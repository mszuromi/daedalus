"""
First-order-in-time guard.

Daedalus integrates Itô SDEs / SPDEs / DAEs.  Their MSR–JD
representation is built on the assumption that the residual is **first
order in the time derivative** ``Dt`` — every term carries at most one
factor of ``Dt``.  Everything downstream bakes that in:

* the mean-field linearization pencil ``(A, B)`` in
  :mod:`api._mean_field_dae` splits the residual into the part at
  ``Dt = 0`` (``B``) and ``∂/∂Dt`` at ``Dt = 0`` (``A``);
* the propagator kernel in :mod:`api._propagator` maps
  ``c0 + c1·Dt  →  c0·δ(t) + c1·δ′(t)``.

A term of ``Dt``-degree ≥ 2 falls through BOTH: it is killed by
``Dt → 0`` in the algebraic part, and its ``∂/∂Dt`` is still
proportional to ``Dt`` so it is killed there too.  It therefore
vanishes from the linearization without a trace, and the run returns a
confidently wrong answer.  (In the propagator it fares no better —
``Dt²`` becomes ``δ′(t)²``, whose Fourier transform Sage renders as
meaningless ``dirac_delta(0)`` factors.)

Rather than silently drop such a term, the pipeline rejects it here,
with a message that spells out the standard first-order rewrite.

Public API
----------
``dt_degree_exceeds_one(expr, dt_sym)``
    Predicate: is ``expr`` nonlinear in ``dt_sym``?
``check_first_order_in_dt(expr, dt_sym, context=..., subject=...)``
    Raise :class:`NonlinearDtError` if so.
``check_action_first_order_in_dt(by_tp, dt_sym, ring_var_names, ...)``
    Same check swept over every monomial of a bigrade-classified action.
"""
from __future__ import annotations

__all__ = [
    'NonlinearDtError',
    'dt_degree_exceeds_one',
    'check_first_order_in_dt',
    'check_action_first_order_in_dt',
]


class NonlinearDtError(ValueError):
    """A model term has degree ≥ 2 in the time-derivative symbol ``Dt``.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    around model parsing keep working, while callers that want to
    single this failure out (and must NOT swallow it) can catch the
    precise type.
    """


_REMEDY = """\
Remedy — rewrite the system in FIRST-ORDER form by promoting each
higher time derivative to its OWN field.  For

    Dt^2*x + a*Dt*x + b*x = 0

declare a second physical field ``v`` (with its own response field)
and split the equation in two:

    .equation(lhs='Dt*x',           rhs='v')       # v := dx/dt
    .equation(lhs='Dt*v + a*v',     rhs='-b*x')    # original eq. in v

mirroring the same split in ``.set_action_text(...)``, e.g.

    xt*(Dt*x - v) + vt*(Dt*v + a*v + b*x) - D*vt^2

This is why the framework requires first-order form.  It costs no generality:
any well-posed system that can be solved for its highest time derivative
reduces to first order exactly this way."""


def _is_zero_expr(e) -> bool:
    """Best-effort ``e == 0`` test for a Sage SR expression.

    Escalates trivial → expanded → fully simplified, so a false
    positive (flagging a genuinely-linear model) needs all three to
    fail.  The expensive step only runs on the would-be-error path.
    """
    try:
        if e.is_trivial_zero():
            return True
    except (AttributeError, TypeError, ValueError):
        pass
    for reducer in ('expand', 'simplify_full'):
        try:
            if getattr(e, reducer)().is_trivial_zero():
                return True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
    return False


def dt_degree_exceeds_one(expr, dt_sym) -> bool:
    """True when ``expr`` has degree ≥ 2 in ``dt_sym``.

    Implemented as "``∂²expr/∂Dt²`` is not identically zero" rather
    than a polynomial ``.degree()`` call, so it also catches
    non-polynomial ``Dt``-dependence (``exp(Dt)``, ``1/(1+Dt)``, …)
    that no polynomial ring would accept in the first place.
    """
    from sage.all import SR, diff

    try:
        e = SR(expr)
    except (TypeError, ValueError, ArithmeticError):
        # Not coercible to SR (plain int / float / ring element with no
        # symbolic content) ⇒ no Dt dependence at all.
        return False
    try:
        if not e.has(SR(dt_sym)):
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    try:
        d2 = diff(diff(e, dt_sym), dt_sym)
    except (AttributeError, TypeError, ValueError):
        return False
    return not _is_zero_expr(d2)


def check_first_order_in_dt(expr, dt_sym, *, context: str, subject: str,
                            dt_name: str = 'Dt',
                            display_subs: dict = None) -> None:
    """Raise :class:`NonlinearDtError` if ``expr`` is nonlinear in ``Dt``.

    Parameters
    ----------
    expr : Sage SR expression (or anything coercible)
        The residual / action coefficient to test.
    dt_sym : Sage SR variable
        The time-derivative symbol used inside ``expr``.
    context : str
        Where the offending expression lives — model name plus the
        declaration it came from.  Appears verbatim in the message.
    subject : str
        The specific equation / field / term at fault.
    dt_name : str
        User-facing spelling of the operator (always ``'Dt'`` today;
        the internal symbol may be a mangled alias).
    display_subs : dict, optional
        ``{internal_symbol_name: user_facing_name}`` applied to the
        rendered expression, so mangled internals (``_mfdae_x_0``)
        surface as the names the user actually wrote (``x``).
    """
    if not dt_degree_exceeds_one(expr, dt_sym):
        return
    from sage.all import SR
    shown = str(SR(expr))
    # Longest-first so ``_mfdae_x_10`` isn't clobbered by ``_mfdae_x_1``.
    for internal, nice in sorted((display_subs or {}).items(),
                                 key=lambda kv: -len(kv[0])):
        shown = shown.replace(internal, nice)
    shown = shown.replace(str(dt_sym), dt_name)
    raise NonlinearDtError(
        f"Model is NONLINEAR in the time derivative {dt_name} "
        f"(degree >= 2).\n"
        f"  where : {context}\n"
        f"  term  : {subject}\n"
        f"  expr  : {shown}\n"
        f"\n"
        f"Daedalus models Ito SDEs/SPDEs/DAEs, whose MSR-JD "
        f"representation must be FIRST ORDER in {dt_name} — at most one "
        f"factor of {dt_name} per term.  A {dt_name}^2 term has no "
        f"first-order image: it is killed by {dt_name} -> 0 in the "
        f"algebraic part of the linearization AND by {dt_name} -> 0 in "
        f"the d/d{dt_name} part, so it would drop out of the pencil "
        f"(and out of the propagator kernel) with no trace, returning a "
        f"confidently wrong answer.  Hence this hard error instead.\n"
        f"\n{_REMEDY}"
    )


def _monomial_text(exp_vec, ring_var_names) -> str:
    """Render an exponent tuple as ``xt1*dx1^2`` for the error message."""
    parts = []
    for name, e in zip(ring_var_names, exp_vec):
        if e == 1:
            parts.append(str(name))
        elif e > 1:
            parts.append(f'{name}^{e}')
    return '*'.join(parts) if parts else '1'


def check_action_first_order_in_dt(by_tp, dt_sym, ring_var_names, *,
                                   model_name: str = '<unnamed>',
                                   origin: str = 'set_action_text(...)') -> None:
    """Sweep a bigrade-classified action for ``Dt``-degree ≥ 2.

    ``by_tp`` is ``FieldTheory._by_tp``: ``{(n_tilde, n_phys): poly}``
    over ``PolynomialRing(SR, ring_var_names)``, so every ``Dt`` lives
    in a monomial's SR *coefficient*.  Cheap in the common case — the
    ``.has(Dt)`` pre-filter skips the (vast) majority of coefficients
    before any differentiation happens.
    """
    from sage.all import SR

    if dt_sym is None or not by_tp:
        return
    dt_sr = SR(dt_sym)
    names = list(ring_var_names)
    for (n_t, n_p), poly in sorted(by_tp.items()):
        try:
            terms = poly.dict()
        except (AttributeError, TypeError):
            continue
        for exp_vec, coeff in terms.items():
            c = SR(coeff)
            try:
                if not c.has(dt_sr):
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            check_first_order_in_dt(
                c, dt_sym,
                context=(f"model {model_name!r}, action declared via "
                         f"{origin}"),
                subject=(f"action sector (n_tilde={n_t}, n_phys={n_p}), "
                         f"monomial {_monomial_text(exp_vec, names)}"),
            )

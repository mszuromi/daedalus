"""
msrjd.core.convolution
======================
Formal convolution operator for time-domain field theory.

Motivation
----------
When a registered kernel ``g`` (i.e. an entry in ``model['kernels']``) is
multiplied by a registered field ``n`` (an entry in ``model['physical_fields']``
or its response counterpart) inside an action / equation expression, the
``*`` syntactically reads as scalar multiplication but **semantically
means convolution**::

    (g * n)(t) ≡ (g ⋆ n)(t) = ∫ g(t − s) · n(s) ds

The convolution evaluates differently depending on what downstream code
is doing with the expression:

  * **Stationary mean field** — at a constant saddle, the integral picks
    up the kernel's DC gain ``ĝ(0) = ∫g(t)dt``.  For normalised kernels
    (``ĝ(0) = 1``) the convolution reduces to the bare field value::

         Conv(g, n*) → ĝ(0) · n*   (= n* for normalised kernels)

  * **Fourier-domain propagator** — by the convolution theorem::

         Conv(g, n)(t) → ĝ(ω) · n̂(ω)

  * **Time-domain simulator** — the convolution is realised as an
    explicit filter state, with one ODE state per pole of the kernel's
    rational Fourier image.  E.g. for the alpha kernel
    ``α(t) = (t/τ²) e^{-t/τ} H(t)``, ``ĝ(ω) = 1/(1+iωτ)²`` has a double
    pole at ``ω = i/τ``, so the simulator integrates two cascaded
    first-order filters (``F_aux`` and ``F``) instead of computing the
    integral directly.

To make the dispatch unambiguous, the theory builder rewrites every
kernel-field product inside an expression into the formal Sage function
:func:`Conv`.  Each downstream consumer then registers its own
reduction rule (substitution, FT, state-space realisation) and applies
it where appropriate.  The convolution operator itself stays inert
under normal Sage symbolic manipulations so the substitution chain can
defer the decision to whichever consumer is doing the work.

Status
------
Step 1 of the equation-first refactor: ``Conv`` is exported here and
threaded into ``_build_namespace_for_eval`` so user expressions can
already reference it.  Auto-rewriting of ``g * n`` patterns to
``Conv(g, n)`` lives in a later step (the ``set_equation`` builder
method).  Until that lands, ``Conv`` is available but inert — existing
``set_action_text`` theories keep their current ``g * n``-as-scalar-
multiplication semantics, with the kernel's frequency image substituted
in via ``model['kernel_ft_image']`` after Fourier transform of the
action.
"""
from __future__ import annotations

from sage.all import function as _sr_function


# Two-argument formal Sage function.  By convention: ``Conv(kernel, field)``
# — kernel first, field second.  The function never auto-evaluates; it stays
# as a formal symbol through Sage's symbolic manipulation pipeline so each
# downstream consumer can perform its own reduction.
#
# ``nargs=2`` lets Sage validate call arity; misuse like ``Conv(g)`` or
# ``Conv(g, n, extra)`` raises immediately rather than producing a silently
# wrong symbolic expression.
Conv = _sr_function('Conv', nargs=2)


def is_convolution(expr) -> bool:
    """Return ``True`` if ``expr`` is a ``Conv(...,...)`` atom.

    Used by symbolic visitors that walk an action / equation tree and need
    to dispatch on convolution terms vs. ordinary products.  Tolerant of
    non-symbolic inputs (returns ``False``).
    """
    try:
        op = expr.operator()
    except (AttributeError, ValueError, TypeError):
        return False
    if op is None:
        return False
    # Sage's `function('Conv', ...)` round-trips through str(op) as 'Conv'.
    return str(op) == 'Conv'


def kernel_of(expr):
    """Return the kernel argument of a ``Conv(kernel, field)`` expression.

    Raises ``ValueError`` if ``expr`` is not a ``Conv`` atom.
    """
    if not is_convolution(expr):
        raise ValueError(f"not a Conv expression: {expr!r}")
    return expr.operands()[0]


def field_of(expr):
    """Return the field argument of a ``Conv(kernel, field)`` expression.

    Raises ``ValueError`` if ``expr`` is not a ``Conv`` atom.
    """
    if not is_convolution(expr):
        raise ValueError(f"not a Conv expression: {expr!r}")
    return expr.operands()[1]


def reduce_conv_in_action(expr, fluct_vars, normalized=True):
    """Resolve every ``Conv(g, n_arg)`` atom in an action expression,
    applying the physical rules of time-domain convolution:

      1. **Linearity in argument 2**:
         ``Conv(g, a + b) → Conv(g, a) + Conv(g, b)``
      2. **Pull constant factors out**:
         ``Conv(g, c·X) → c · Conv(g, X)``  when ``c`` is independent
         of the fluctuation-field variables.  (Convolution is over time;
         time-independent prefactors come out.)
      3. **DC reduction for constants**:
         ``Conv(g, c) → ĝ(0) · c``  (= ``c`` for normalised kernels).
         A constant function of time convolves with the kernel to give
         the kernel's integral times the constant.
      4. **Defer fluctuations to the FT pipeline**:
         ``Conv(g, X) → g · X``  when ``X`` carries genuine fluctuation
         dependence.  The kernel SR symbol ``g`` (entries of
         ``model['kernels']``) is replaced by ``ĝ(ω)`` later via
         ``kernel_ft_image``, which is the convolution theorem.

    The asymmetry of the convolution operator — only argument 2 (the
    field) carries the time-domain content; argument 1 (the kernel) is
    a time-translation-invariant filter — is preserved.  In particular
    a conductance-style term ``v · Conv(g, n)`` expands correctly under
    field substitution ``v = vstar + dv, n = nstar + dn``::

        v · Conv(g, n)
        = (vstar + dv) · (Conv(g, nstar) + Conv(g, dn))
        = (vstar + dv) · (nstar + g·dn)                   [rules 1, 3, 4]
        = vstar·nstar + vstar·g·dn + dv·nstar + dv·g·dn

    Whereas the naïve ``Conv → g·n`` flatten would have given
    ``v·g·n = vstar·g·nstar + …`` which spuriously couples the dv-side
    of the bilinear to the kernel.

    Parameters
    ----------
    expr : SR expression
        Action (or any expression) containing ``Conv(.,.)`` atoms.
    fluct_vars : iterable of SR symbols
        The fluctuation-field generators (``dv``, ``dn``, ``vt``,
        ``nt``, ...).  Anything else — including saddle parameters
        like ``vstar``, ``nstar`` and external parameters like ``a``,
        ``w``, ``tau`` — is treated as time-constant.
    normalized : bool, default True
        Assume every kernel has ``ĝ(0) = 1``.  Set ``False`` and
        supply per-kernel DC gains in a future revision when
        non-normalised kernels become relevant.

    Returns
    -------
    SR expression with all ``Conv`` atoms resolved according to the
    rules above.  Kernel SR symbols remain in fluctuation-dependent
    pieces so the existing Fourier-transform pipeline can substitute
    ``ĝ(ω)`` for them.
    """
    from functools import reduce as _reduce
    from sage.all import prod as _prod, SR as _SR

    fluct_set = set(fluct_vars) if not isinstance(fluct_vars, (set, frozenset)) \
                                else fluct_vars

    def _has_fluct(e):
        try:
            return bool(set(_SR(e).variables()) & fluct_set)
        except (AttributeError, TypeError):
            return False

    def _reduce_arg(g, n_arg):
        """Reduce Conv(g, n_arg).  Recurses on sums and products; falls
        through to the per-atom resolution at the leaves."""
        n_arg = _SR(n_arg).expand()
        op = n_arg.operator() if hasattr(n_arg, 'operator') else None
        op_name = (str(op).lower() if op is not None else '')

        # Rule 1: distribute over sums.
        if 'add' in op_name:
            return sum(_reduce_arg(g, term) for term in n_arg.operands())

        # Rule 2: pull constant factors out of products.
        if 'mul' in op_name:
            factors = list(n_arg.operands())
            const_factors = [f for f in factors if not _has_fluct(f)]
            fluct_factors = [f for f in factors if     _has_fluct(f)]

            if not fluct_factors:
                # Whole product is constant ⇒ rule 3.
                return _prod(const_factors) if const_factors else _SR(1)

            const_part = (_reduce(lambda a, b: a * b, const_factors)
                          if const_factors else _SR(1))
            fluct_part = _reduce(lambda a, b: a * b, fluct_factors)
            # const_part · Conv(g, fluct_part)
            return const_part * _reduce_arg(g, fluct_part)

        # Leaf — atom or unhandled operator.
        if not _has_fluct(n_arg):
            return _SR(n_arg)              # rule 3
        return g * n_arg                   # rule 4

    return expr.substitute_function(Conv, _reduce_arg)


__all__ = ['Conv', 'is_convolution', 'kernel_of', 'field_of',
           'reduce_conv_in_action']

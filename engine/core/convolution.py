"""
engine.core.convolution
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

To make the dispatch unambiguous, the model builder rewrites every
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
``set_action_text`` models keep their current ``g * n``-as-scalar-
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


class KernelDCGains:
    """The integral ``ĝ(0) = ∫g(t)dt`` of each declared kernel, read off
    the model's ``kernel_ft_image`` hook at ``omega = 0`` (so a delay factor
    ``exp(-I*omega*d)`` counts 1 and a Pade copy of it P(0)/Q(0) = 1).

    Lazy: the hook runs once, on the first request (a ``time_expr`` kernel
    costs a symbolic Fourier transform), and only kernels actually met at a
    constant are evaluated.  A kernel without an image (``DeltaKernel``, an
    opaque symbol) has no gain here (``None``): the caller keeps the
    normalised rule ``Conv(g, c) = c`` for it.  A gain equal to 1 comes back
    as the exact ``SR(1)``, so a normalised kernel leaves every expression
    untouched."""

    def __init__(self, ns, model):
        self._ns = ns
        self._hook = (model or {}).get('kernel_ft_image')
        self._images = None
        self._omega = None
        self._sym_cache = {}
        # the kernel symbols on the namespace (no hook call): an expression
        # without any of them never runs the hook
        self._ksyms = set()
        for kspec in (model or {}).get('kernels', []) or []:
            self._ksyms.update(_flat_symbols(getattr(ns, kspec['name'], None)))

    def _mentions_kernel(self, expr):
        from sage.all import SR as _SR
        if not self._ksyms or self._hook is None:
            return False
        try:
            return any(v in self._ksyms for v in _SR(expr).variables())
        except Exception:
            return False

    def _load(self):
        if self._images is not None:
            return
        from sage.all import SR as _SR
        self._images = {}
        if self._hook is None:
            return
        om = _SR.var('_kernel_dc_omega')
        try:
            imgs = self._hook(self._ns, om) or {}
        except Exception as exc:
            # an image the hook cannot build fails later, where the
            # propagator needs it, with the hook's own message; say here
            # that the saddle took the kernels as normalised meanwhile
            import warnings as _w
            _w.warn(f'kernel images could not be built ({exc}); the '
                    f'mean-field saddle takes every kernel as normalised '
                    f'(integral 1).', stacklevel=3)
            return
        self._omega = om
        self._images = {_SR(k): _SR(v) for k, v in imgs.items()}

    def symbols(self):
        """The kernel symbols that have an image."""
        self._load()
        return list(self._images)

    def symbol_gain(self, ksym):
        """``ĝ(0)`` of one kernel symbol, or ``None`` without an image.
        Raises ``ValueError`` when the image is singular at omega = 0 (a
        kernel with no finite integral, e.g. a pure integrator)."""
        from sage.all import SR as _SR
        self._load()
        ksym = _SR(ksym)
        if ksym in self._sym_cache:
            return self._sym_cache[ksym]
        img = self._images.get(ksym)
        if img is None:
            self._sym_cache[ksym] = None
            return None
        def _finite(x):
            txt = str(x)
            return not ('Infinity' in txt or 'infinity' in txt
                        or 'NaN' in txt or 'und' == txt or 'ind' == txt)
        try:
            g0 = img.subs({self._omega: 0})
            bad = not _finite(g0)
        except Exception:
            bad = True
        if bad:
            # a removable singularity (a boxcar (1 - exp(-I*omega*T))/(I*omega*T),
            # sin(omega*T)/(omega*T)) has a finite limit; a pole does not
            try:
                from sage.all import limit as _limit
                g0 = _SR(_limit(img, **{str(self._omega): 0}))
                bad = not _finite(g0) or g0.has(self._omega)
            except Exception:
                bad = True
        if bad:
            raise ValueError(
                f'kernel {ksym}: its frequency image {img} (omega = '
                f'{self._omega}) is singular at omega = 0, so the kernel '
                f'has no finite integral and Conv({ksym}, constant) has no '
                f'stationary value; the mean-field saddle does not exist.')
        if not g0.is_trivially_equal(_SR(1)):
            try:
                if (g0 - 1).simplify_full().is_zero():
                    g0 = _SR(1)
            except Exception:
                pass
        else:
            g0 = _SR(1)
        self._sym_cache[ksym] = g0
        return g0

    def gain(self, kexpr):
        """``ĝ(0)`` of a kernel argument of ``Conv`` (a kernel symbol, or an
        expression in kernel symbols such as ``2*g``), or ``None`` when it
        holds no kernel with an image."""
        from sage.all import SR as _SR
        if not self._mentions_kernel(kexpr):
            return None
        self._load()
        if not self._images:
            return None
        try:
            kexpr = _SR(kexpr)
            ks = [v for v in kexpr.variables() if v in self._images]
        except Exception:
            return None
        if not ks:
            return None
        subs = {v: self.symbol_gain(v) for v in ks}
        g = kexpr.subs(subs)
        return _SR(1) if g.is_trivially_equal(_SR(1)) else g

    def subs_dict(self, expr):
        """``{kernel symbol: ĝ(0)}`` for the kernel symbols with an image
        that occur in ``expr`` (empty when none do)."""
        from sage.all import SR as _SR
        if not self._mentions_kernel(expr):
            return {}
        self._load()
        if not self._images:
            return {}
        try:
            present = [v for v in _SR(expr).variables() if v in self._images]
        except Exception:
            return {}
        return {v: self.symbol_gain(v) for v in present}


def _flat_symbols(val):
    """The SR symbols of a namespace kernel entry (a symbol, a list, or a
    list of lists of them)."""
    from sage.all import SR as _SR
    if isinstance(val, (list, tuple)):
        for v in val:
            yield from _flat_symbols(v)
    elif val is not None:
        try:
            if _SR(val).is_symbol():
                yield _SR(val)
        except Exception:
            return


def reduce_conv_in_action(expr, fluct_vars, normalized=True, taylor_order=None,
                          attachments_out=None, dc_gains=None):
    """Resolve every ``Conv(g, n_arg)`` atom in an action expression,
    applying the physical rules of time-domain convolution.

    Processing order for each ``Conv(g, n_arg)`` atom:

      0. **Pre-Taylor inside the argument** (when ``taylor_order`` given):
         Multivariate Taylor-expand ``n_arg`` around 0 in every
         fluctuation variable, truncating at total degree
         ``taylor_order``.  This is what turns a nonlinear argument
         like ``phi(v)`` into a polynomial in ``v`` that the subsequent
         linearity rule can distribute over.  No-op when ``n_arg`` is
         already a polynomial of degree ≤ taylor_order in the fluct
         vars (e.g. a bare ``n[j]``), so existing callers that never
         use nonlinear Conv args see identical output.

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

    The pre-Taylor step (rule 0) generalises this to nonlinear inner
    fields ``Conv(g, h(v, n))``.  For example with cubic ``h(v) = a v^3``,
    a saddle expansion ``v = vstar + dv`` gives::

        h(vstar+dv) = a vstar^3 + 3 a vstar^2 dv + 3 a vstar dv^2 + a dv^3

    and applying rules 1–4 produces::

        Conv(g, h(vstar+dv))
        = a vstar^3 + 3 a vstar^2 g dv + 3 a vstar g dv^2 + a g dv^3

    Each Taylor term is then handled by the existing linearity/pull-
    constants/DC-reduce/defer cascade.

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
        Unused (kept for callers); see ``dc_gains``.
    dc_gains : KernelDCGains, optional
        The kernels' integrals ``ĝ(0)`` for rule 3.  Without it (or for a
        kernel it has no image of) every kernel is taken as normalised,
        ``ĝ(0) = 1``.
    taylor_order : int, optional
        Total-degree truncation for the pre-Taylor step (rule 0).
        Set to ``FieldTheory.taylor_order`` by the caller to keep the
        Conv-internal expansion consistent with the action-level
        expansion that follows downstream.  When ``None``, the
        pre-Taylor step is skipped — preserves the legacy semantics
        for any caller that doesn't supply it.
    attachments_out : dict, optional
        Out-parameter: when supplied, the reducer records every rule-4
        emission as a key→value pair ``kernel_symbol → set of leg
        variables`` the kernel got paired with.  Used downstream by
        vertex extraction to identify which leg each surviving kernel
        symbol in an interaction-vertex coefficient is attached to
        (the leg whose frequency ``ĝ(ω_leg)`` should be evaluated at
        in Fourier space).  Same kernel reused with different fields
        accumulates: ``Conv(g, n1) + Conv(g, n2)`` yields
        ``attachments_out[g] = {n1, n2}``.  The caller is responsible
        for disambiguating per-vertex by index when a kernel attaches
        to multiple fields.  The dict is mutated in place; pre-existing
        entries are augmented (set-union), not overwritten.

    Returns
    -------
    SR expression with all ``Conv`` atoms resolved according to the
    rules above.  Kernel SR symbols remain in fluctuation-dependent
    pieces so the existing Fourier-transform pipeline can substitute
    ``ĝ(ω)`` for them.
    """
    from functools import reduce as _reduce
    from sage.all import prod as _prod, SR as _SR, taylor as _taylor

    fluct_set = set(fluct_vars) if not isinstance(fluct_vars, (set, frozenset)) \
                                else fluct_vars
    # Pre-build the Taylor expansion-point pairs once, reused for every
    # Conv atom.  Each entry is (var, 0) — i.e. expand around the
    # saddle, which lives at fluctuation = 0 by convention.
    _taylor_pairs = (tuple((v, 0) for v in fluct_set)
                     if taylor_order is not None and fluct_set else None)

    def _has_fluct(e):
        try:
            return bool(set(_SR(e).variables()) & fluct_set)
        except (AttributeError, TypeError):
            return False

    def _dc(g, c):
        """Rule 3: ``Conv(g, c) = ĝ(0)·c``; unchanged for ĝ(0) = 1."""
        if dc_gains is None or _SR(c).is_trivial_zero():
            return c
        gain = dc_gains.gain(g)
        if gain is None or gain.is_trivially_equal(_SR(1)):
            return c
        return gain * c

    def _reduce_arg(g, n_arg):
        """Reduce Conv(g, n_arg).  Pre-Taylor (rule 0) first, then
        recurses on sums and products; falls through to the per-atom
        resolution at the leaves."""
        # Rule 0: pre-Taylor the Conv argument in fluctuation variables.
        # Converts nonlinear ``h(v, n)`` into a polynomial so the
        # linearity rule (rule 1) can distribute over the resulting
        # sum.  Multivariate one-shot Taylor at total degree
        # ``taylor_order`` — same call pattern that
        # ``FieldTheory.expand`` uses on the whole action a few lines
        # downstream, so the orders stay in sync.
        if _taylor_pairs is not None:
            try:
                n_arg = _taylor(_SR(n_arg), *_taylor_pairs, taylor_order)
            except (TypeError, ValueError, AttributeError):
                # Non-Taylor-expandable argument (e.g. piecewise);
                # fall through to the existing rules unchanged.
                pass

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
                return _dc(g, _prod(const_factors) if const_factors
                           else _SR(1))

            const_part = (_reduce(lambda a, b: a * b, const_factors)
                          if const_factors else _SR(1))
            fluct_part = _reduce(lambda a, b: a * b, fluct_factors)
            # const_part · Conv(g, fluct_part)
            return const_part * _reduce_arg(g, fluct_part)

        # Leaf — atom or unhandled operator.
        if not _has_fluct(n_arg):
            return _dc(g, _SR(n_arg))      # rule 3
        # Rule 4: defer the kernel to the FT pipeline.  Record the
        # (kernel, attached-fluct) association so downstream vertex
        # extraction can recover which leg the kernel goes with.
        if attachments_out is not None:
            try:
                attached_flucts = set(_SR(n_arg).variables()) & fluct_set
            except (AttributeError, TypeError):
                attached_flucts = set()
            if attached_flucts:
                # Same kernel can pair with several fields across the
                # action — set-union accumulates them.
                bucket = attachments_out.setdefault(g, set())
                bucket.update(attached_flucts)
        return g * n_arg                   # rule 4

    return expr.substitute_function(Conv, _reduce_arg)


def kernel_attachments_in_coefficient(coeff, attachments, kernel_symbols=None):
    """Identify which kernel symbols appear in ``coeff`` and the
    fluctuation field each is attached to.

    Designed for downstream vertex extraction: after bigrade
    classification, each interaction-vertex coefficient is an SR
    expression that may contain kernel SR symbols (from earlier
    ``Conv(g, fluct) → g · fluct`` rule-4 emissions).  This helper
    looks up the kernel-attachment record collected by
    ``reduce_conv_in_action(..., attachments_out=attachments)`` and
    returns, for each kernel symbol present in ``coeff``, the leg
    variable it was attached to.

    Parameters
    ----------
    coeff : SR expression
        A vertex coefficient (the symbolic prefactor of a monomial in
        the bigrade-classified action).
    attachments : dict
        Populated by ``reduce_conv_in_action``'s ``attachments_out``
        parameter.  Maps kernel SR symbol → set of attached fluct vars.
    kernel_symbols : iterable of SR symbols, optional
        Restrict the scan to these symbols.  Defaults to all keys of
        ``attachments``.  Useful when the same kernel SR variable
        could appear in the action for reasons other than a Conv
        rule-4 (e.g. a direct ``g * x`` product the user wrote
        explicitly), in which case the caller can supply the model's
        ``kernel`` symbol list to avoid false positives.

    Returns
    -------
    dict {kernel_symbol: leg_var}.  When a kernel symbol attaches to
    a single fluct var, the value is that var.  When the same kernel
    attaches to multiple legs (a kernel reused across the action),
    the value is a ``frozenset`` of the candidate leg vars — the
    caller must disambiguate via index matching against the vertex's
    actual legs.  Kernel symbols not present in ``coeff`` are omitted.
    """
    from sage.all import SR as _SR
    try:
        coeff_vars = set(_SR(coeff).variables())
    except (AttributeError, TypeError):
        return {}
    if kernel_symbols is None:
        kernel_symbols = list(attachments.keys())
    out = {}
    for ksym in kernel_symbols:
        if ksym not in coeff_vars:
            continue
        leg_set = attachments.get(ksym)
        if not leg_set:
            continue
        if len(leg_set) == 1:
            out[ksym] = next(iter(leg_set))
        else:
            out[ksym] = frozenset(leg_set)
    return out


__all__ = ['KernelDCGains', 'Conv', 'is_convolution', 'kernel_of', 'field_of',
           'reduce_conv_in_action', 'kernel_attachments_in_coefficient']

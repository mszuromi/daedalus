"""
pipeline.theory_compiler — text expressions → Sage lambdas.

When a ``TheoryBuilder`` is fed text expressions (via
``set_action_text``, ``define_function``, ``define_kernel``,
``set_mf_equation``, ``declare_cgf_term``), this module turns each
text string into a callable that the FieldTheory expander can use.

Design
------
We don't pre-parse anything at ``set_*`` time.  Each lambda factory
in this module returns a closure that, when invoked with the
runtime ``ns`` (a NamespaceForFields), assembles a Sage-eval
namespace from ``ns``'s attributes plus declared parameters /
functions / kernels, then ``sage_eval`` s the stored text.

Parsing happens at lambda-call time.  Expressions are small and
the number of calls is low (a handful per ``compute_cumulants``
invocation), so this is cheap.  Caching the parsed AST is a
future optimization.

Index conventions
-----------------
Action and MF equations are written **per-population** with ``i`` as
the implicit free index — ``S_i`` and ``vstar[i] = ...``.  At
build time the closure substitutes ``i`` with each concrete
population index ``0..N_pop-1`` and sums (action) or stores
per-saddle (MF equations).

Inner sums use Python comprehension syntax::

    sum(w[i, j] * g * dn[j] for j in pop)

where ``pop = range(N_pop)`` is pre-bound in the namespace.
"""
from __future__ import annotations

from typing import Any, Callable

from sage.all import (
    SR, sage_eval, exp, log, sin, cos, tan, sqrt,
    heaviside, dirac_delta, I, pi, function as sr_function, diff,
)


# ── Helpers ───────────────────────────────────────────────────────────

class _IndexedFormalFunction:
    """Generates Sage formal function calls indexed by population.

    Usage in compiler-bound action namespace::

        phi = _IndexedFormalFunction('phi')
        phi[i](dv[i])     # → function(f'phi_{i+1}')(dv[i])
        phi[i, j](dv[i])  # → function(f'phi_{i+1}_{j+1}')(dv[i])

    Each indexed call returns a Python callable so the user can then
    apply it to the field argument.  This supports both scalar (one
    transfer function for all populations, distinguished by index for
    auto-Taylor) and per-pair (e.g., synaptic-pair-specific) variants
    when the function is declared with explicit indexing.
    """
    __slots__ = ('_name',)

    def __init__(self, name: str):
        self._name = name

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            sfx = '_'.join(str(int(k) + 1) for k in idx)
        else:
            sfx = str(int(idx) + 1)
        full_name = f'{self._name}_{sfx}'
        def _call(arg, _n=full_name):
            return sr_function(_n)(arg)
        return _call


class _IndexedSaddleRename:
    """Maps ``f(arg)`` and ``f[i](arg)`` directly to the formal
    Taylor-rename target ``SR.var(f'{name}0_<i+1>')`` instead of going
    through Sage's symbolic Taylor expansion.

    Used in mf_bg saddle-substitution evaluation: when the user writes
    ``set_mf_equation('nstar', 'phi(vstar[i])')``, we want mf_bg to
    return ``{nstar[i]: phi0_<i+1>}`` so the action's ``nt[i] * nstar[i]``
    cancels symbolically against the Taylor-expanded
    ``nt[i] * phi0_<i+1>`` term from the Poisson cumulant.
    """
    __slots__ = ('_name',)

    def __init__(self, name: str):
        self._name = name

    def __call__(self, arg, i=None):
        # Bare-call form: phi(vstar[i]).  Caller binds ``i`` via the
        # outer for-loop in mf_bg evaluation.
        if i is None:
            # Fallback: try to infer from arg's variable name if it
            # ends in a digit (e.g. vstar1 → 0).  Conservative.
            try:
                s = str(arg)
                # Last digits in the variable name
                import re
                m = re.search(r'(\d+)\s*$', s)
                if m:
                    i = int(m.group(1)) - 1
                else:
                    i = 0
            except Exception:
                i = 0
        return SR.var(f'{self._name}0_{int(i) + 1}')

    def __getitem__(self, idx):
        # Indexed form: phi[i](vstar[i]) — directly returns the
        # rename target for this i.
        if isinstance(idx, tuple):
            sfx = '_'.join(str(int(k) + 1) for k in idx)
        else:
            sfx = str(int(idx) + 1)
        target = SR.var(f'{self._name}0_{sfx}')
        def _call(arg, _t=target):
            return _t
        return _call


class _FullPhysicalField:
    """Exposes ``n[i] = nstar[i] + dn[i]`` (full physical field as
    saddle + fluctuation) in user-facing action expressions.

    The user writes the action in physical observables:
      ``nt[i] * n[i]``   instead of   ``nt[i] * (nstar[i] + dn[i])``
      ``phi[i](v[i])``   instead of   ``phi[i](dv[i])``

    The saddle-fluctuation split is handled by the framework: ``n``,
    ``v``, ``m`` etc. are bound to ``_FullPhysicalField`` objects in
    the action namespace, while ``dn``, ``dv``, ``dm`` (pure
    fluctuations) and ``nstar``, ``vstar``, ``mstar`` (pure saddles)
    remain accessible under their internal names for users who
    prefer the explicit form.
    """
    __slots__ = ('_saddle', '_fluct')

    def __init__(self, saddle_array, fluct_array):
        self._saddle = saddle_array
        self._fluct  = fluct_array

    def __getitem__(self, i):
        if isinstance(i, tuple):
            # Chained subscript, e.g. multi-index — generally not
            # used for physical fields, but support gracefully.
            return self._saddle[i] + self._fluct[i]
        return self._saddle[i] + self._fluct[i]

    def __len__(self):
        return len(self._fluct)


class _MatrixView:
    """Wrap a list-of-lists ``rows`` so it accepts both tuple subscript
    ``w[i, j]`` and chained subscript ``w[i][j]``.  Lets users write
    ``w[i, j]`` in Sage-syntax expressions without surprise."""

    __slots__ = ('_rows',)

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self._rows[i][j]
        return self._rows[key]

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __repr__(self):
        return f'_MatrixView({self._rows!r})'


def _builtin_namespace() -> dict[str, Any]:
    """Symbols that every user expression may freely reference."""
    return {
        'exp':             exp,
        'log':             log,
        'sin':             sin,
        'cos':             cos,
        'tan':             tan,
        'sqrt':            sqrt,
        'heaviside':       heaviside,
        'delta_function':  dirac_delta,
        'dirac_delta':     dirac_delta,
        'sum':             sum,
        'I':               I,
        'pi':              pi,
    }


def _ns_var_namespace(ns, field_names, param_names, kernel_names,
                      naming_convention=None,
                      expand_to_full=False) -> dict[str, Any]:
    """Assemble user-name → ns-attribute dict.

    For each declared field/parameter/kernel ``name`` the corresponding
    ``ns.<name>`` is exposed under that name.  When a
    ``naming_convention`` dict is supplied, fields are ALSO exposed
    under their natural names — but the binding depends on
    ``expand_to_full``:

    - ``expand_to_full=False`` (default; MF equations, etc.): natural
      name is an alias for the fluctuation field.  ``n[i]`` returns
      ``ns.dn[i]``.
    - ``expand_to_full=True`` (action context): natural name is bound
      to ``_FullPhysicalField(ns.<saddle>, ns.<fluct>)``.  ``n[i]``
      returns ``nstar[i] + dn[i]`` (the full physical observable).
      The internal names ``dn``, ``dv`` remain accessible for
      explicit fluctuation references.

    Matrix-shaped parameters are wrapped in :class:`_MatrixView` for
    tuple-subscript access.
    """
    out: dict[str, Any] = {}

    # Fields (vectors of SR vars).  Expose under internal name AND
    # under natural name when the model declares a translation.
    fluct_natural_to_internal = (
        (naming_convention or {}).get('fluctuation_fields') or {})
    saddle_natural_to_internal = (
        (naming_convention or {}).get('mean_field_saddles') or {})
    fluct_internal_to_natural = {v: k
                                 for k, v in fluct_natural_to_internal.items()}

    for fname in field_names:
        if not hasattr(ns, fname):
            continue
        val = getattr(ns, fname)
        out[fname] = val
        # Also expose under natural name if there's a mapping.
        # In action context, natural name is the FULL field
        # (saddle + fluctuation); elsewhere it aliases the fluctuation.
        natural = fluct_internal_to_natural.get(fname)
        if natural and natural != fname:
            if expand_to_full:
                saddle_internal = saddle_natural_to_internal.get(natural)
                if saddle_internal and hasattr(ns, saddle_internal):
                    out[natural] = _FullPhysicalField(
                        getattr(ns, saddle_internal), val)
                else:
                    out[natural] = val
            else:
                out[natural] = val

    # Parameters — promote 2D lists to MatrixView for w[i,j] access.
    # Saddle parameters are also exposed under their natural names
    # so users can write nstar[i] OR n_star[i].  We expose under the
    # internal name (nstar) only since the latter is more standard.
    for pname in param_names:
        if not hasattr(ns, pname):
            continue
        val = getattr(ns, pname)
        if isinstance(val, list) and val and isinstance(val[0], list):
            out[pname] = _MatrixView(val)
        else:
            out[pname] = val

    # Kernels (single SR symbols)
    for kname in kernel_names:
        if hasattr(ns, kname):
            out[kname] = getattr(ns, kname)

    # Operators
    if hasattr(ns, 'Dt'):
        out['Dt'] = ns.Dt

    return out


def _build_namespace_for_eval(ns, *, field_names, param_names, kernel_names,
                              functions, n_pop, transfer_function=None,
                              transfer_function_mode='formal', i=None,
                              naming_convention=None, extra=None):
    """Full namespace dict to feed ``sage_eval`` as ``locals=...``.

    Includes built-ins, ns-derived symbols, compiled functions, and
    iteration helpers (``pop = range(n_pop)``).

    The transfer function (typically ``phi``) is bound differently
    depending on context:

    - ``transfer_function_mode='formal'`` — the function returns the
      formal Sage symbol ``function(f'{name}_{i+1}')(arg)`` so that
      FieldTheory's auto-expander Taylor-expands it.  Required in the
      action (so the framework generates ``phi0_<i>``, ``phi1_<i>``,
      ``phi2_<i>``, … vertices).

    - ``transfer_function_mode='concrete'`` — the function evaluates
      the user's text expression and returns the concrete SR
      expression (e.g. ``a * v^2``).  Required in MF equations
      (where ``phi(vstar[i])`` must give a concrete number that the
      saddle-solver can use).
    """
    nsdict = _builtin_namespace()
    nsdict.update(_ns_var_namespace(
        ns, field_names, param_names, kernel_names,
        naming_convention=naming_convention,
        expand_to_full=(transfer_function_mode == 'action'),
    ))
    nsdict['pop'] = list(range(n_pop))

    # Bind every user-defined function.  In the **action** context all
    # functions become formal indexed Sage symbols ('action' mode) so
    # FieldTheory's auto-Taylor pass produces ``<name>0_<i+1>``,
    # ``<name>1_<i+1>``, ... derivative symbols for every function.
    # In other contexts (MF equations, etc.) functions are bound
    # CONCRETELY by default — but a caller can request the formal-rename
    # mode by passing ``transfer_function_mode='mf_formal_rename'``,
    # which makes ``f(arg)`` evaluate to ``SR.var(f'{name}0_<i+1>')``
    # (the formal Taylor-coefficient symbol that mf_bg's saddle-EOM
    # closure substitutes into the action).
    for fn in functions:
        fname = fn['name']
        if transfer_function_mode == 'action':
            nsdict[fname] = _IndexedFormalFunction(fname)
        elif transfer_function_mode == 'mf_formal_rename':
            nsdict[fname] = _IndexedSaddleRename(fname)
        else:    # 'concrete' (default for MF eq evaluation)
            nsdict[fname] = _make_function_callable(fn, nsdict)

    # Backward-compatible: if a caller still passes a legacy
    # ``transfer_function``, leave the existing binding in place (it's
    # already covered by the loop above when the function is in
    # ``functions``).

    if extra:
        nsdict.update(extra)
    return nsdict


def _make_function_callable(fn_spec: dict, parent_ns: dict) -> Callable:
    """Turn a ``define_function`` spec into a Python callable.

    ``fn_spec = {'name': 'phi', 'args': ['v'], 'expression': 'a*v^2'}``
    yields a function ``phi(v_value)`` that returns the SR expression
    ``a * v_value^2`` (with ``a`` resolved against the parent namespace).
    """
    args = fn_spec['args']
    expr_text = fn_spec['expression']

    def _callable(*arg_values):
        if len(arg_values) != len(args):
            raise TypeError(
                f'{fn_spec["name"]}() expects {len(args)} args ({args}); '
                f'got {len(arg_values)}'
            )
        local_ns = dict(parent_ns)
        for argname, argval in zip(args, arg_values):
            local_ns[argname] = argval
        return sage_eval(expr_text, locals=local_ns)

    _callable.__name__ = fn_spec['name']
    return _callable


def _normalize_expr_text(text: str) -> str:
    """Prep a user-typed expression for ``sage_eval``.

    Triple-quoted strings in Python source carry their indentation; the
    Python parser inside ``sage_eval`` rejects leading whitespace.  We
    dedent, strip surrounding blank lines, and join physical lines into
    a single logical expression by collapsing newlines (since the user
    expressions are arithmetic, line breaks just split a long sum).
    """
    import textwrap
    text = textwrap.dedent(text).strip()
    # Collapse newlines + tabs into a single space so ``a + b\n  + c``
    # parses as one expression.  Sage syntax doesn't depend on
    # newlines, so this is safe.
    text = ' '.join(text.split())
    return text


def _safe_eval(text: str, locals_dict: dict, what: str) -> Any:
    """Evaluate user-typed Sage-syntax ``text`` against ``locals_dict``.

    Two things ``sage_eval`` would normally do are reproduced manually
    here so we can use plain ``eval`` (necessary for the generator-
    expression scoping fix below):

    1. ``preparse`` rewrites Sage syntax — most importantly turns
       ``a*v^2`` (which is XOR in Python) into ``a*v**2``, and lifts
       integer literals to Sage Integers.
    2. The namespace is passed as **globals** rather than **locals**
       so that generator expressions inside e.g. ``sum(... for j in
       pop)`` can see ``w``, ``dn``, etc.  ``eval``'s ``locals`` arg
       only feeds the outermost scope.

    The combined globals dict is ``sage.all.__dict__`` ⊔ ``locals_dict``,
    so user-supplied names override Sage builtins where they clash
    (e.g. user might have a parameter named ``I`` shadowing the
    imaginary unit).
    """
    import sage.all as _sage_all
    from sage.repl.preparse import preparse
    text = _normalize_expr_text(text)
    parsed = preparse(text)
    eval_globals = {**_sage_all.__dict__, **locals_dict}
    try:
        return eval(parsed, eval_globals)
    except Exception as e:
        raise ValueError(
            f'Could not parse {what}:\n  {text!r}\n'
            f'  (preparsed: {parsed!r})\n\n'
            f'Error: {e}\n\n'
            f'Available symbols: {sorted(locals_dict.keys())}'
        ) from e


# ── Lambda factories (one per model hook) ─────────────────────────────

def make_action_lambda(action_text: str, *, field_names, param_names,
                       kernel_names, functions, n_pop,
                       transfer_function=None,
                       naming_convention=None):
    """Build the ``model['action']`` lambda from the user's action text.

    The action is evaluated as a single Sage expression — user writes
    the **full** action including all sums.  Population iteration
    uses Python comprehension against pre-bound ``pop``::

        sum(nt[i] * n[i] - (exp(nt[i]) - 1) * phi[i](v[i])
            + vt[i] * (...)
            for i in pop)

    Field references are FULL physical observables: ``n[i]`` resolves
    to ``nstar[i] + dn[i]`` automatically, and ``phi[i](v[i])`` works
    correctly because the Taylor-rename map is augmented to include
    saddle-point expansions (``phi(vstar[i] + dv[i])`` Taylor-expands
    in ``dv[i]`` around 0, producing terms like
    ``function('phi_<i+1>')(vstar[i])``, which we map to the same
    ``phi0_<i+1>`` / ``phi1_<i+1>`` / ... formal symbols that
    ``mf_bg_conditions`` and ``specializations`` substitute).
    """
    def _action(ns):
        nsdict = _build_namespace_for_eval(
            ns,
            field_names = field_names,
            param_names = param_names,
            kernel_names = kernel_names,
            functions   = functions,
            n_pop       = n_pop,
            transfer_function = transfer_function,
            transfer_function_mode = 'action',
            naming_convention = naming_convention,
        )
        s = _safe_eval(action_text, nsdict, 'action')

        # Kill Dt * <saddle>[i] * (anything) terms — saddle quantities
        # are time-constant by definition.  Necessary because the user
        # writes the action in physical fields (n[i] = nstar[i] + dn[i],
        # v[i] = vstar[i] + dv[i]); operators like (tau*Dt + 1) * v[i]
        # expand to tau*Dt*vstar[i] + tau*Dt*dv[i] + vstar[i] + dv[i],
        # and the framework needs ``tau*Dt*vstar[i] = 0`` enforced.
        # Done here (in the action lambda, BEFORE field_theory.py's
        # mf_bg substitution of vstar) so the kill rules match the raw
        # vstar1, vstar2, ... symbols rather than their saddle-equation
        # expansions.
        if hasattr(ns, 'Dt'):
            saddle_internals = (
                (naming_convention or {}).get('mf_parameters') or [])
            if saddle_internals:
                W = SR.wild()
                kill: dict = {}
                for sname in saddle_internals:
                    if not hasattr(ns, sname):
                        continue
                    arr = getattr(ns, sname)
                    if not isinstance(arr, (list, tuple)):
                        continue
                    for elem in arr:
                        kill[ns.Dt * elem]     = SR(0)
                        kill[ns.Dt * elem * W] = SR(0)
                if kill:
                    # Sage's subs with Mul-pattern matching is more
                    # reliable on the EXPANDED form (each term laid out
                    # as an explicit Mul rather than nested in
                    # parentheses).  Expand → subs → no-op if already
                    # expanded.
                    s = SR(s).expand().subs(kill)

        # Augment ns._deriv_rename_subs with saddle-point Taylor renames
        # for EVERY declared function.  This makes
        # ``f[i](v[i])`` (where v[i] = vstar[i] + dv[i]) Taylor-expand
        # cleanly: the formal terms produced by Sage's taylor() at
        # vstar[i] (rather than at 0) get mapped to the framework's
        # ``f<k>_<i+1>`` rename targets.
        if hasattr(ns, '_deriv_rename_subs'):
            _augment_saddle_renames(
                ns, functions,
                naming_convention=naming_convention,
                taylor_order=getattr(ns, '_taylor_order', 4),
            )

        return s

    return _action


def _augment_saddle_renames(ns, functions, *,
                            naming_convention=None,
                            taylor_order=4):
    """Add saddle-point Taylor-rename entries to ``ns._deriv_rename_subs``
    for **every** declared function.

    For each function ``f`` (any user-chosen name) and each population
    ``i``, registers::

        function('f_<i+1>')(<saddle>[i])      → f0_<i+1>
        D[0](function('f_<i+1>'))(<saddle>[i]) → f1_<i+1>
        D[0,0](...)(<saddle>[i])               → f2_<i+1>
        ...

    where ``<saddle>`` is the saddle of the function's first argument
    (looked up via the model's naming_convention).  This means writing
    ``f[i](v[i])`` in the action — where ``v[i]`` expands to
    ``vstar[i] + dv[i]`` — Taylor-expands cleanly via the framework's
    rename machinery, regardless of the user's chosen function name.
    """
    if not functions:
        return
    saddle_map = (naming_convention or {}).get('mean_field_saddles') or {}
    x_dum = SR.var('_xdum_saddle_')

    for fn_spec in functions:
        tname = fn_spec['name']
        arg_names = fn_spec.get('args') or []
        if not arg_names:
            continue
        # Saddle for this function's first arg.
        arg_natural = arg_names[0]
        saddle_internal = saddle_map.get(arg_natural, f'{arg_natural}star')
        if not hasattr(ns, saddle_internal):
            continue
        saddle_array = getattr(ns, saddle_internal)

        for i in range(len(ns.pop)):
            fe = sr_function(f'{tname}_{i + 1}')(x_dum)
            for k in range(taylor_order + 1):
                if k == 0:
                    val = fe.subs({x_dum: saddle_array[i]})
                else:
                    val = diff(fe, x_dum, k).subs({x_dum: saddle_array[i]})
                target = SR.var(f'{tname}{k}_{i + 1}')
                ns._deriv_rename_subs[val] = target


def make_phi_concrete_lambda(phi_fn_spec: dict, *, field_names, param_names,
                             kernel_names, functions, n_pop):
    """Build ``model['phi_concrete']`` from a single user-declared
    transfer function.

    The pipeline-internal MF code calls ``phi_concrete(ns, i, v_sym)``
    where ``v_sym`` is a fresh symbolic variable used to take Taylor
    derivatives.  We bind the function's first argument to ``v_sym``
    (regardless of what the user named it) and evaluate the body.
    """
    expr_text = phi_fn_spec['expression']
    arg_names = phi_fn_spec['args']
    if len(arg_names) != 1:
        raise ValueError(
            f"transfer function must take exactly 1 argument; "
            f"{phi_fn_spec['name']!r} takes {len(arg_names)} ({arg_names})"
        )
    arg_name = arg_names[0]

    def _phi_concrete(ns, i, v_sym):
        base_ns = _build_namespace_for_eval(
            ns,
            field_names = field_names,
            param_names = param_names,
            kernel_names = kernel_names,
            functions   = [],   # phi can only reference parameters, not other functions
            n_pop       = n_pop,
        )
        return _safe_eval(
            expr_text, {**base_ns, arg_name: v_sym, 'i': i},
            f'transfer function {phi_fn_spec["name"]}({arg_name})')

    return _phi_concrete


def make_specializations_lambda(phi_fn_spec: dict | None, *,
                                field_names, param_names,
                                kernel_names, functions, n_pop,
                                naming_convention=None,
                                mf_eqs: dict[str, str] = None):
    # Note: ``mf_eqs`` parameter retained for API compatibility but
    # no longer used — closure substitution is handled by mf_bg.
    """Auto-derive ``model['specializations']`` for every declared
    function.

    For each function ``f`` (any name) and each population ``i``,
    produces:

      * ``f<k>_<i+1>  →  f^(k)(<saddle>[i])`` for k = 1, 2, ..., order
        — the kth derivative of f's concrete expression evaluated at
        the saddle of f's first arg.

      * ``f0_<i+1>``  →  one of two forms:

        - **closure form** (``f0_<i+1> → ns.<closure_saddle>[i]``):
          if the user declared ``set_mf_equation('<saddle>',
          'f(<arg>)')``, then f is the saddle-EOM closure for
          ``<saddle>``.  The framework substitutes ``f0`` with the
          ``<saddle>`` iteration variable, ensuring the (1, 0) tadpole
          on the response field cancels symbolically.

        - **concrete form** (``f0_<i+1> → f(<saddle>[i])``):
          if no mf_equation closes f, substitute the literal value
          of f at the saddle.

    The function name is just a label — ``phi``, ``f``, ``response_fn``
    all behave identically.
    """
    saddle_map = (naming_convention or {}).get('mean_field_saddles') or {}

    def _specs(ns):
        base_ns = _build_namespace_for_eval(
            ns,
            field_names = field_names,
            param_names = param_names,
            kernel_names = kernel_names,
            functions   = [],
            n_pop       = n_pop,
        )
        out: dict = {}
        order = ns._taylor_order

        for fn_spec in (functions or []):
            tname = fn_spec['name']
            arg_names = fn_spec.get('args') or []
            if not arg_names:
                continue
            arg_name = arg_names[0]
            saddle_internal = saddle_map.get(arg_name,
                                             f'{arg_name}star')
            if not hasattr(ns, saddle_internal):
                continue
            saddle_array = getattr(ns, saddle_internal)

            # Pre-compute Taylor-coefficient values at the saddle.
            v_sym = SR.var(f'_{tname}_taylor_arg')
            f_at_v = _safe_eval(
                fn_spec['expression'],
                {**base_ns, arg_name: v_sym},
                f'{tname}({arg_name}) Taylor expansion')

            # Always-concrete substitution: f<k>_<i+1> → kth derivative
            # of f's concrete expression evaluated at the saddle.
            # k=0 gives ``f(<saddle>[i])``, k=1 gives
            # ``f'(<saddle>[i])``, etc.  No saddle-name shortcut —
            # the action-side mf_bg has already substituted the
            # iteration saddle with its formal-rename target, so the
            # tadpole cancellation works without specs needing to
            # know about closures.
            for i in range(n_pop):
                for k in range(0, order + 1):
                    sym = SR.var(f'{tname}{k}_{i+1}')
                    if k == 0:
                        out[sym] = f_at_v.subs(
                            {v_sym: saddle_array[i]})
                    else:
                        deriv_at_v = f_at_v.derivative(v_sym, k)
                        out[sym] = deriv_at_v.subs(
                            {v_sym: saddle_array[i]})
        return out

    return _specs


_SINGLE_CALL_RE = None   # lazy-compiled below


def _is_single_function_call(rhs_text: str) -> bool:
    """True iff the RHS text is exactly one function call ``f(arg)`` —
    no surrounding arithmetic.  Used to detect saddle-EOM closures
    (e.g. ``nstar = phi(vstar[i])``) so the framework can substitute
    the iteration saddle with the formal Taylor-rename target
    ``<func>0_<i+1>`` instead of the concrete derivative value.

    Returns False for compound expressions like ``a*phi(v) + b`` or
    ``E + sum(w*g*nstar)``.
    """
    global _SINGLE_CALL_RE
    if _SINGLE_CALL_RE is None:
        import re
        # Match: optional whitespace, identifier, optional [i] / [i,j],
        # opening paren, anything balanced, closing paren, optional ws.
        # Crude check: balanced parens by character count.
        _SINGLE_CALL_RE = re.compile(
            r'^\s*\w+\s*(\[[^\]]+\]\s*)?\((.+)\)\s*$', re.DOTALL)
    text = (rhs_text or '').strip()
    m = _SINGLE_CALL_RE.match(text)
    if not m:
        return False
    # Confirm parens are balanced AT THE TOP level — else 'a*f(x) + b'
    # could be matched due to outer parens of the whole expression.
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _classify_mf_eqs(mf_eqs: dict[str, str], iteration_saddle: str) -> dict:
    """Split mf_eqs into (closure_saddles, compound_saddles).

    * **closure saddles**: their RHS expresses the saddle in terms of
      function calls (e.g. ``nstar = phi(v) + b``).  These get
      substituted in the action via formal-rename evaluation
      (``nstar → phi0_<i+1> + b``) — the formal-rename target gets
      Taylor-expanded by the framework.

    * **compound saddles**: their RHS is a parameter / saddle
      expression (e.g. ``vstar = E + sum(w·g·nstar)``).  These get
      substituted concretely.

    The default convention treats the iteration_saddle (typically
    ``'nstar'``) as a closure saddle if its mf_eq is present, since
    that's the saddle the solver iterates on while everything else
    follows from it.  Any other saddle is compound.
    """
    closure = {}
    compound = {}
    for saddle_name, rhs_text in mf_eqs.items():
        if saddle_name == iteration_saddle:
            closure[saddle_name] = rhs_text
        else:
            compound[saddle_name] = rhs_text
    return {'closure': closure, 'compound': compound}


def make_mf_bg_conditions_lambda(mf_eqs: dict[str, str],
                                 phi_fn_spec: dict | None,
                                 *,
                                 field_names, param_names, kernel_names,
                                 functions, n_pop,
                                 iteration_saddle: str = 'nstar'):
    """Build the **action-side** ``mf_bg_conditions`` lambda.

    Two-pass design that handles both simple Hawkes-style closures
    (``nstar = phi(v)``) and compound forms (``nstar = phi(v) + b``)
    uniformly:

      1. **Closure pass**: the iteration saddle's mf_eq RHS is
         evaluated in *formal-rename* mode (function calls return the
         formal Taylor-coefficient target ``f0_<i+1>``).  Result for
         compound: ``nstar[i] → phi0_<i+1> + b``.

      2. **Compound pass**: every other saddle's mf_eq RHS is
         evaluated concretely, then the closure substitution is
         applied to the result.  E.g., ``vstar[i] → E[i] + sum(w·g·
         nstar[j])`` becomes ``vstar[i] → E[i] + sum(w·g·(phi0_<j+1>
         + b))`` after baking in the closure.

    Both substitutions get applied to the symbolic action with
    Sage single-pass subs.  Because vstar's RHS no longer has raw
    nstar (it's been pre-baked with the closure), the (1, 0) tadpole
    cancels symbolically regardless of the saddle EOM form.

    The companion :func:`make_mf_bg_solver_lambda` builds the
    SOLVER-friendly version (raw concrete, no closure).
    """
    classification = _classify_mf_eqs(mf_eqs, iteration_saddle)
    closure_eqs  = classification['closure']
    compound_eqs = classification['compound']

    def _bg(ns):
        # ── Pass 1: closure saddles (formal-rename eval) ──────────
        # nstar (or whichever the iteration saddle is) gets
        # substituted with the formal-rename eval of its mf_eq RHS.
        # Works for any RHS form:
        #   set_mf_equation('nstar', 'phi(vstar[i])')      → phi0_<i+1>
        #   set_mf_equation('nstar', 'phi(vstar[i]) + b')  → phi0_<i+1>+b
        #   set_mf_equation('nstar', 'a*phi(vstar[i])^2')  → a*phi0_<i+1>^2
        closure_subs: dict = {}
        for saddle_name, rhs_text in closure_eqs.items():
            if not hasattr(ns, saddle_name):
                continue
            saddle_array = getattr(ns, saddle_name)
            for i in range(n_pop):
                ns_i = _build_namespace_for_eval(
                    ns,
                    field_names  = field_names,
                    param_names  = param_names,
                    kernel_names = kernel_names,
                    functions    = functions,
                    n_pop        = n_pop,
                    transfer_function = phi_fn_spec,
                    transfer_function_mode = 'mf_formal_rename',
                    i           = i,
                )
                rhs = _safe_eval(
                    rhs_text, {**ns_i, 'i': i},
                    f'MF equation {saddle_name}[i] (closure)')
                closure_subs[saddle_array[i]] = rhs

        # ── Pass 2: compound saddles (concrete eval, closure-baked) ─
        # vstar (and any other compound saddle) is evaluated concretely.
        # Then the closure substitutions are applied to the result so
        # raw nstar references inside vstar's RHS get replaced with
        # the formal-rename closure form (preventing the single-pass-
        # subs problem in field_theory's symbolic action substitution).
        out: dict = dict(closure_subs)
        for saddle_name, rhs_text in compound_eqs.items():
            if not hasattr(ns, saddle_name):
                continue
            saddle_array = getattr(ns, saddle_name)
            for i in range(n_pop):
                ns_i = _build_namespace_for_eval(
                    ns,
                    field_names  = field_names,
                    param_names  = param_names,
                    kernel_names = kernel_names,
                    functions    = functions,
                    n_pop        = n_pop,
                    transfer_function = phi_fn_spec,
                    transfer_function_mode = 'concrete',
                    i           = i,
                )
                rhs = _safe_eval(
                    rhs_text, {**ns_i, 'i': i},
                    f'MF equation {saddle_name}[i] (compound)')
                if closure_subs:
                    rhs = SR(rhs).subs(closure_subs)
                out[saddle_array[i]] = rhs

        # Dt * <saddle>[i] → 0  (saddle quantities are time-constant
        # by definition).  Necessary because the user writes the action
        # in physical fields ``v[i] = vstar[i] + dv[i]``; multiplying
        # by a kinetic operator ``(tau*Dt + 1) * v[i]`` produces a
        # spurious ``tau*Dt*vstar[i]`` term that the framework must
        # zero out.  We use a wild-card subs so factors like
        # ``tau * Dt * vstar1`` (Mul of multiple terms) are captured
        # alongside the bare two-factor case.
        if hasattr(ns, 'Dt'):
            W = SR.wild()
            for saddle_name in mf_eqs.keys():
                if not hasattr(ns, saddle_name):
                    continue
                arr = getattr(ns, saddle_name)
                for i in range(n_pop):
                    out[ns.Dt * arr[i]]     = SR(0)
                    out[ns.Dt * arr[i] * W] = SR(0)
        return out

    return _bg


def make_mf_bg_solver_lambda(mf_eqs: dict[str, str],
                             phi_fn_spec: dict | None,
                             *,
                             field_names, param_names, kernel_names,
                             functions, n_pop,
                             iteration_saddle: str = 'nstar'):
    """Build the **solver-friendly** ``mf_bg_conditions`` lambda.

    Returns a dict with EVERY saddle entry (both compound and
    iteration) in raw concrete form (with all other saddles still
    as their raw SR symbols).  The iteration saddle's entry is the
    user's mf_eq RHS evaluated concretely — e.g. for
    ``set_mf_equation('nstar', 'phi(vstar) + b')`` the entry is
    ``ns.nstar[i] → a*ns.vstar[i]**2 + b``.  ``solve_mean_field``
    reads this entry and uses it as the iteration target.

    Concretely::

        # Hawkes simple:
        out[ns.vstar[i]] = E[i] + sum(w[i,j]*g*ns.nstar[j])
        out[ns.nstar[i]] = a * ns.vstar[i]**2

        # Compound (nstar = phi(v) + b):
        out[ns.nstar[i]] = a * ns.vstar[i]**2 + b
    """
    def _bg_solver(ns):
        out: dict = {}
        for saddle_name, rhs_text in mf_eqs.items():
            if not hasattr(ns, saddle_name):
                continue
            saddle_array = getattr(ns, saddle_name)
            for i in range(n_pop):
                ns_i = _build_namespace_for_eval(
                    ns,
                    field_names  = field_names,
                    param_names  = param_names,
                    kernel_names = kernel_names,
                    functions    = functions,
                    n_pop        = n_pop,
                    transfer_function = phi_fn_spec,
                    transfer_function_mode = 'concrete',
                    i           = i,
                )
                rhs = _safe_eval(
                    rhs_text, {**ns_i, 'i': i},
                    f'MF equation {saddle_name}[i] (solver)')
                out[saddle_array[i]] = rhs
        return out

    return _bg_solver


def make_mf_equations_lambda(mf_eqs: dict[str, str],
                             phi_fn_spec: dict | None,
                             *, field_names, param_names, kernel_names,
                             functions, n_pop):
    """Build ``model['mf_equations']`` — the residuals to solve.

    Returns a list of equations ``ns.<saddle>[i] == <rhs>`` for each
    saddle/i pair.  ``solve_mean_field`` doesn't actually use this
    (it solves numerically via fsolve), but the FieldTheory expander
    keeps it for symbolic consistency checks.
    """
    def _eqs(ns):
        out = []
        for saddle_name, rhs_text in mf_eqs.items():
            if not hasattr(ns, saddle_name):
                continue
            saddle_array = getattr(ns, saddle_name)
            for i in range(n_pop):
                ns_i = _build_namespace_for_eval(
                    ns,
                    field_names = field_names,
                    param_names = param_names,
                    kernel_names = kernel_names,
                    functions   = functions,
                    n_pop       = n_pop,
                    transfer_function = phi_fn_spec,
                    transfer_function_mode = 'concrete',
                    i           = i,
                )
                rhs = _safe_eval(
                    rhs_text, {**ns_i, 'i': i},
                    f'MF equation {saddle_name}[i]')
                out.append(saddle_array[i] == rhs)
        return out

    return _eqs


def make_kernel_ft_image_lambda(kernel_specs: list[dict], *, param_names):
    """Build ``model['kernel_ft_image']`` from the kernel declarations.

    Each kernel spec may carry either:
      - ``'freq_image'`` : text expression in ``omega`` and parameters
                           — used directly.
      - ``'time_expr'``  : text expression in ``t`` and parameters
                           — Fourier-transformed symbolically at
                           build time (best-effort).
      - neither          : skipped (kernel stays as opaque symbol;
                           e.g. δ-kernels handled by DeltaKernel).

    Returns a lambda ``(ns, omega) -> {ns.<kname>: SR_expr_in_omega}``.
    """
    def _ft_image(ns, omega):
        out: dict = {}
        param_ns = _ns_var_namespace(ns, [], param_names, [])
        builtins = _builtin_namespace()
        eval_ns = {**builtins, **param_ns, 'omega': omega}

        for spec in kernel_specs:
            kname = spec['name']
            if not hasattr(ns, kname):
                continue
            ksym = getattr(ns, kname)
            if spec.get('freq_image'):
                out[ksym] = _safe_eval(
                    spec['freq_image'], eval_ns,
                    f"kernel {kname}'s freq_image")
            # time_expr → freq image: deferred (would need Sage's
            # fourier_transform on each).  Only freq_image supported
            # for v1 — TheoryBuilder warns when only time_expr given.

        return out

    return _ft_image


def make_correlated_noises_block(cgf_terms: list[dict], *, param_names):
    """Convert declared CGF cumulant terms into the
    ``model['correlated_noises']`` dict consumed by FieldTheory.

    ``cgf_terms`` is a list of dicts::

        {'name': 'X', 'response_field': 'mt', 'order': 2,
         'coefficient': 'lambda_X * p_part * (1 + sigma_shift_diff_sq)',
         'kernel': None}        # optional time-domain kernel string

    Multiple rows can share name+response — they sum into the same
    cumulant, possibly with different kernels.

    Returns a dict ready to drop into ``model['correlated_noises']``.
    The values are themselves dicts with per-order entries that the
    FieldTheory framework's NoiseSourceType machinery consumes at
    expansion time.
    """
    out: dict[str, dict] = {}
    for term in cgf_terms:
        name = term['name']
        if name not in out:
            out[name] = {
                'response_field': term['response_field'],
                'cumulants':      {},   # order → coefficient_text
            }
        order = int(term['order'])
        if order in out[name]['cumulants']:
            # Combine: sum the coefficient text expressions
            existing = out[name]['cumulants'][order]
            out[name]['cumulants'][order] = (
                f'({existing}) + ({term["coefficient"]})')
        else:
            out[name]['cumulants'][order] = term['coefficient']
        if term.get('kernel'):
            out[name].setdefault('kernels', {})[order] = term['kernel']
    return out

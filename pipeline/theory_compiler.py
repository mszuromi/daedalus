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
        phi[i](dv[i])           # → function(f'phi_{i+1}')(dv[i])
        phi[i, j](dv[i])        # → function(f'phi_{i+1}_{j+1}')(dv[i])
        f[i](v[i], n[i])        # → function(f'f_{i+1}')(v[i], n[i])

    Each indexed call returns a variadic callable so the user can then
    apply the formal function to one OR multiple field arguments.
    Multi-arg formal calls are picked up by Sage's multivariate
    ``taylor()`` in :func:`field_theory.expand` — every partial
    derivative ``∂^α f / ∂x_1^{α_1} ... ∂x_n^{α_n}`` at the saddle
    expansion point gets renamed to ``f<α_1>...<α_n>_<i+1>`` by the
    framework's auto-Taylor pass.
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
        def _call(*args, _n=full_name):
            return sr_function(_n)(*args)
        return _call


class _IndexedSaddleRename:
    """Maps formal function calls in mf_bg evaluation directly to the
    framework's Taylor-rename target — bypassing Sage's symbolic
    expansion — so the action-side closure substitution lines up with
    the bigrade pass's symbolic vertex names.

    Single-arg example::

        set_mf_equation('nstar', 'phi(vstar[i])')
        # mf_bg returns {nstar[i]: phi0_<i+1>}, which cancels the
        # action's nt[i]*nstar[i] tadpole against the Poisson
        # Taylor's nt[i]*phi0_<i+1>.

    Multi-arg example (now supported)::

        set_mf_equation('nstar', 'f(vstar[i], nstar[i])')
        # mf_bg returns {nstar[i]: f00_<i+1>} — the zeroth multi-
        # derivative of the 2-arg formal f.  Multi-arg auto-Taylor
        # in FieldTheory.expand() produces the same f00_<i+1>
        # symbol, so the tadpole cancellation still works.

    ``n_args`` (default 1) controls the trailing-zero count in the
    rename suffix — `'0' * n_args`.  For single-arg the suffix is
    ``0`` (legacy behavior); for n-arg it is ``00...0`` (n times).
    Multi-arg calls that don't match ``n_args`` exactly fall back
    to whatever ``len(args)`` is at call time, since that's the
    authoritative count from the user's mf_eq syntax.
    """
    __slots__ = ('_name', '_n_args')

    def __init__(self, name: str, n_args: int = 1):
        self._name = name
        self._n_args = max(int(n_args), 1)

    def _build_target(self, n_args: int, i: int) -> 'SR':
        suffix = '0' * max(n_args, 1)
        return SR.var(f'{self._name}{suffix}_{int(i) + 1}')

    def __call__(self, *args, i=None):
        # Bare-call form: phi(vstar[i]) — or for multi-arg,
        # f(vstar[i], nstar[i]).  Caller binds ``i`` via the
        # outer for-loop in mf_bg evaluation.
        n_args = len(args) if args else self._n_args
        if i is None:
            # Fallback: try to infer from the first arg's variable
            # name if it ends in a digit (e.g. vstar1 → 0).
            try:
                s = str(args[0]) if args else ''
                import re
                m = re.search(r'(\d+)\s*$', s)
                if m:
                    i = int(m.group(1)) - 1
                else:
                    i = 0
            except Exception:
                i = 0
        return self._build_target(n_args, i)

    def __getitem__(self, idx):
        # Indexed form: phi[i](vstar[i]) — or, for multi-arg,
        # f[i](vstar[i], nstar[i]).  Returns a variadic callable
        # so it accepts however many args the user's mf_eq passes.
        if isinstance(idx, tuple):
            sfx_pop = '_'.join(str(int(k) + 1) for k in idx)
        else:
            sfx_pop = str(int(idx) + 1)
        name      = self._name
        default_n = self._n_args
        def _call(*args, _name=name, _pop=sfx_pop, _default_n=default_n):
            n_args = len(args) if args else _default_n
            suffix = '0' * max(n_args, 1)
            return SR.var(f'{_name}{suffix}_{_pop}')
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

    # Kernels can be scalar, vector (list), or matrix (list-of-lists).
    # Wrap list-of-lists in _MatrixView so ``g[i, j]`` tuple subscript
    # works the same way it does for matrix-valued parameters.
    for kname in kernel_names:
        if hasattr(ns, kname):
            val = getattr(ns, kname)
            if isinstance(val, list) and val and isinstance(val[0], list):
                out[kname] = _MatrixView(val)
            else:
                out[kname] = val

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
    # Iteration ranges.
    #   * ``pop`` — legacy single-population flat index, always bound.
    #   * ``pop_<name>`` — per-population index, e.g. ``pop_E``.
    #   * ``<name>``      — plain-name alias, e.g. ``E`` ↔ pop_E.  This
    #     lets heterogeneous-pop action text say ``for i in E``
    #     instead of ``for i in pop_E``.  Plain aliases shadow any
    #     equally-named parameter — users should pick non-conflicting
    #     population names.
    nsdict['pop'] = list(range(n_pop))
    pop_local_idx = getattr(ns, '_pop_local_idx', {}) or {}
    for pname, plist in pop_local_idx.items():
        nsdict[f'pop_{pname}'] = list(plist)
        if pname not in nsdict:
            nsdict[pname] = list(plist)

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
            # Thread the function's declared arity through so the
            # rename emits the right suffix length (``f0_<i+1>`` for
            # single-arg, ``f00_<i+1>`` for 2-arg, etc.).  The runtime
            # call's actual arg count overrides this if it differs,
            # which makes the rename robust to either bare or indexed
            # call syntax in the user's mf_eq.
            nsdict[fname] = _IndexedSaddleRename(
                fname, n_args=len(fn.get('args') or []) or 1)
        else:    # 'concrete' (default for MF eq evaluation)
            nsdict[fname] = _make_function_callable(fn, nsdict)

    # Backward-compatible: if a caller still passes a legacy
    # ``transfer_function``, leave the existing binding in place (it's
    # already covered by the loop above when the function is in
    # ``functions``).

    if extra:
        nsdict.update(extra)
    return nsdict


class _IndexableCallable:
    """Concrete function exposing both ``phi(v)`` and ``phi[i](v)``
    syntax.  In heterogeneous-population theories the function body
    may reference per-population parameters like ``a[i] * v^2``, so
    the ``[i]`` on the call site has to propagate the index into the
    eval scope.

    Behavior:
      * ``phi[i]``  →  returns a new wrapper carrying ``fixed_i = i``.
      * ``phi(v)``  →  evaluates the expression body with ``v``
        bound to the formal argument and (if set) ``i = fixed_i``
        added to the eval namespace.  Single-index ``[i]`` and the
        tuple form ``[i, j]`` are both supported.
    """
    __slots__ = ('_name', '_args', '_expr_text', '_parent_ns', '_fixed_idx')

    def __init__(self, fn_spec, parent_ns, fixed_idx=None):
        self._name      = fn_spec['name']
        self._args      = list(fn_spec.get('args') or [])
        self._expr_text = fn_spec['expression']
        self._parent_ns = parent_ns
        self._fixed_idx = fixed_idx       # None / int / tuple

    def __getitem__(self, idx):
        return _IndexableCallable(
            {'name': self._name, 'args': self._args,
             'expression': self._expr_text},
            self._parent_ns, fixed_idx=idx)

    def __call__(self, *arg_values):
        if len(arg_values) != len(self._args):
            raise TypeError(
                f'{self._name}() expects {len(self._args)} args '
                f'({self._args}); got {len(arg_values)}'
            )
        local_ns = dict(self._parent_ns)
        for argname, argval in zip(self._args, arg_values):
            local_ns[argname] = argval
        # Bind the indexed call-site's population index(es) into the
        # function-body's eval scope so expressions like ``a[i] * v``
        # resolve correctly.
        if self._fixed_idx is not None:
            if isinstance(self._fixed_idx, tuple):
                # phi[i, j] — bind both axes by position.
                if len(self._fixed_idx) >= 1:
                    local_ns['i'] = int(self._fixed_idx[0])
                if len(self._fixed_idx) >= 2:
                    local_ns['j'] = int(self._fixed_idx[1])
            else:
                local_ns['i'] = int(self._fixed_idx)
        return sage_eval(self._expr_text, locals=local_ns)

    def __repr__(self):
        return f'<_IndexableCallable {self._name}({", ".join(self._args)})>'


def _make_function_callable(fn_spec: dict, parent_ns: dict):
    """Turn a ``define_function`` spec into a callable that supports
    both ``f(v)`` and ``f[i](v)`` syntax."""
    return _IndexableCallable(fn_spec, parent_ns)


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


def _iter_multi_indices(n_args, max_total):
    """Yield every multi-index (k_1, ..., k_{n_args}) of non-negative
    integers with  sum k_j  ≤ ``max_total``.

    Iterative implementation (via ``itertools.product``) so it is
    safe under ``%autoreload`` — a self-recursive generator would
    fail to find itself in module globals after a hot-swap.
    """
    if n_args == 0:
        yield ()
        return
    from itertools import product
    for combo in product(range(max_total + 1), repeat=n_args):
        if sum(combo) <= max_total:
            yield combo


def _augment_saddle_renames(ns, functions, *,
                            naming_convention=None,
                            taylor_order=4):
    """Add saddle-point Taylor-rename entries to ``ns._deriv_rename_subs``
    for **every** declared function, including multi-argument ones.

    For an n-argument formal function ``f(arg_1, ..., arg_n)`` with
    saddle expansion point ``(saddle_1[i], ..., saddle_n[i])``, every
    multi-derivative

        ∂^|α| f  /  ∂arg_1^{α_1} ... ∂arg_n^{α_n}    at the saddle

    with ``sum α_j ≤ taylor_order`` is registered as a rename to
    ``SR.var(f'<tname><α_1><α_2>...<α_n>_<i+1>')``.  For single-arg
    functions (n=1) this collapses to the legacy ``<tname><k>_<i+1>``
    naming, so existing models are unaffected.

    Saddles for each argument are looked up via
    ``naming_convention['mean_field_saddles']``, with fallback to
    ``<arg>star``.  An argument with no saddle aborts that function's
    rename registration (its saddle expansion point is undefined).
    """
    if not functions:
        return
    saddle_map = (naming_convention or {}).get('mean_field_saddles') or {}

    pop_size_map = getattr(ns, '_pop_size', {}) or {}

    for fn_spec in functions:
        tname     = fn_spec['name']
        arg_names = fn_spec.get('args') or []
        if not arg_names:
            continue
        n_args = len(arg_names)

        # Resolve every argument's saddle array.
        saddle_arrays = []
        skip = False
        for arg_natural in arg_names:
            saddle_internal = saddle_map.get(arg_natural,
                                             f'{arg_natural}star')
            if not hasattr(ns, saddle_internal):
                skip = True
                break
            saddle_arrays.append(getattr(ns, saddle_internal))
        if skip:
            continue

        # One dummy SR var per formal argument.
        arg_dums = [SR.var(f'_xdum_saddle_{tname}_{j}')
                    for j in range(n_args)]

        # Function index range: prefer the function's declared
        # population (heterogeneous-pop path), else the smallest
        # saddle array (= per-arg population), else legacy flat pop.
        fn_pop = fn_spec.get('population')
        if fn_pop and fn_pop in pop_size_map:
            n_indices = pop_size_map[fn_pop]
        else:
            n_indices = min(len(a) for a in saddle_arrays)

        for i in range(n_indices):
            fe = sr_function(f'{tname}_{i + 1}')(*arg_dums)
            saddle_subs = {arg_dums[j]: saddle_arrays[j][i]
                           for j in range(n_args)}
            for multi_idx in _iter_multi_indices(n_args, taylor_order):
                deriv = fe
                for j, kj in enumerate(multi_idx):
                    if kj > 0:
                        deriv = diff(deriv, arg_dums[j], kj)
                val = deriv.subs(saddle_subs)
                suffix = ''.join(str(k) for k in multi_idx)
                target = SR.var(f'{tname}{suffix}_{i + 1}')
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
    function, including multi-argument ones.

    For an n-arg formal function ``f(arg_1, ..., arg_n)`` and each
    population ``i``, produces

        f<α_1><α_2>...<α_n>_<i+1>
            →  ∂^|α| f / (∂arg_1^{α_1} ... ∂arg_n^{α_n})
                       evaluated at  (saddle_1[i], ..., saddle_n[i])

    for every multi-index α with sum ≤ ``taylor_order``.  Single-arg
    functions (n=1) collapse to the legacy ``f<k>_<i+1>`` naming, so
    existing models are unaffected.

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
            tname     = fn_spec['name']
            arg_names = fn_spec.get('args') or []
            if not arg_names:
                continue
            n_args = len(arg_names)

            # Resolve every argument's saddle array.  An argument with
            # no declared saddle aborts that function's spec emission.
            saddle_arrays = []
            skip = False
            for arg_name in arg_names:
                saddle_internal = saddle_map.get(arg_name,
                                                 f'{arg_name}star')
                if not hasattr(ns, saddle_internal):
                    skip = True
                    break
                saddle_arrays.append(getattr(ns, saddle_internal))
            if skip:
                continue

            # Per-population eval of the function body — so the
            # expression can reference indexed parameters via ``a[i]``,
            # giving each population its own concrete derivatives at
            # the saddle.  ``i`` is bound in the eval namespace below.
            arg_syms = [SR.var(f'_{tname}_taylor_arg_{j}')
                        for j in range(n_args)]

            # Iterate over the saddle's actual size, not ``n_pop``.
            # ``n_pop`` counts how many POPULATIONS the model has;
            # ``len(saddle_arrays[0])`` is the SIZE of the function's
            # own population (the neuron count it indexes over).  For
            # single-pop theories with size > 1 the two differ —
            # ``n_pop=1`` would emit specializations only for
            # ``phi<k>_1`` (neuron 1) and silently skip ``phi<k>_2``
            # etc., leaving those formal symbols unsubstituted in K_ft
            # and producing 0 candidate roots downstream.  Same root
            # cause as the equivalent fix in
            # ``make_mf_bg_conditions_lambda`` / friends.
            n_indices = len(saddle_arrays[0]) if saddle_arrays else n_pop
            for i in range(n_indices):
                local_ns = dict(base_ns)
                for arg_name, arg_sym in zip(arg_names, arg_syms):
                    local_ns[arg_name] = arg_sym
                # Make the population index available inside the
                # expression: users can write ``a[i] * v^2`` to pick the
                # i-th component of an indexed parameter, or any other
                # i-dependent algebra (e.g. ``E[i] + sum(w[i,j]*... )``).
                local_ns['i'] = i
                f_at_args = _safe_eval(
                    fn_spec['expression'], local_ns,
                    f'{tname}({", ".join(arg_names)}) '
                    f'expansion at pop {i+1}')
                saddle_subs = {arg_syms[j]: saddle_arrays[j][i]
                               for j in range(n_args)}
                for multi_idx in _iter_multi_indices(n_args, order):
                    deriv = f_at_args
                    for j, kj in enumerate(multi_idx):
                        if kj > 0:
                            deriv = deriv.derivative(arg_syms[j], kj)
                    suffix = ''.join(str(k) for k in multi_idx)
                    sym = SR.var(f'{tname}{suffix}_{i+1}')
                    out[sym] = deriv.subs(saddle_subs)
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


def _classify_mf_eqs(mf_eqs: dict[str, str], iteration_saddle) -> dict:
    """Split mf_eqs into (closure_saddles, compound_saddles).

    * **closure saddles**: their RHS expresses the saddle in terms of
      function calls (e.g. ``nstar = phi(v) + b``).  These get
      substituted in the action via formal-rename evaluation
      (``nstar → phi0_<i+1> + b``) — the formal-rename target gets
      Taylor-expanded by the framework.

    * **compound saddles**: their RHS is a parameter / saddle
      expression (e.g. ``vstar = E + sum(w·g·nstar)``).  These get
      substituted concretely.

    ``iteration_saddle`` may be:
      * a single name (legacy, single-pop):  ``'nstar'``.
      * a set / list of names (heterogeneous-pop):
        ``{'nEstar', 'nIstar'}`` — each is a closure saddle.
      * the sentinel ``'AUTO'`` — every mf_eq whose RHS is a single
        function call is treated as a closure saddle (the rest are
        compound).  Recommended for theories with more than one
        iteration saddle.
    """
    if iteration_saddle == 'AUTO':
        iter_set = {name for name, rhs in mf_eqs.items()
                    if _is_single_function_call(rhs or '')}
    elif isinstance(iteration_saddle, (set, list, tuple)):
        iter_set = set(iteration_saddle)
    else:
        iter_set = {iteration_saddle}
    closure = {}
    compound = {}
    for saddle_name, rhs_text in mf_eqs.items():
        if saddle_name in iter_set:
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
            # Iterate over the saddle's own size (== population size for
            # heterogeneous-pop theories), not ``n_pop`` (which counts
            # how many populations there are).  For single-pop theories
            # with size > 1 the two differ — ``n_pop=1`` would skip
            # every saddle index past the first.
            for i in range(len(saddle_array)):
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
            for i in range(len(saddle_array)):
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
                for i in range(len(arr)):
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
            for i in range(len(saddle_array)):
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
            for i in range(len(saddle_array)):
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
                           — used directly (fast path).
      - ``'time_expr'``  : text expression in ``t`` and parameters
                           — Fourier-transformed symbolically via
                           ``msrjd.core.field_theory.fourier_transform``
                           at build time.  This is the path the UI
                           uses when the user only specifies a
                           time-domain kernel.
      - neither          : skipped (kernel stays as opaque symbol;
                           e.g. δ-kernels handled by DeltaKernel).

    Returns a lambda ``(ns, omega) -> {ns.<kname>: SR_expr_in_omega}``.
    """
    from msrjd.core.field_theory import fourier_transform as _ft

    def _compute_image(spec, eval_ns, omega, t_var, where):
        """Either evaluate the user-supplied freq_image text or
        Fourier-transform the time_expr.  Returns an SR expression
        in omega."""
        freq_text = spec.get('freq_image')
        if freq_text:
            return _safe_eval(freq_text, eval_ns, where)
        time_text = spec.get('time_expr')
        if not time_text:
            return None
        # Add ``t`` to the eval namespace so the time-domain
        # expression can reference it.
        time_eval_ns = {**eval_ns, 't': t_var}
        time_val = _safe_eval(time_text, time_eval_ns,
                              f'{where} (time_expr)')
        # Best-effort FT — Sage's ``integrate`` handles
        # heaviside(t)*exp(-t/tau) etc. without explicit positivity
        # assumptions for typical neural-kernel forms.
        try:
            img = _ft(time_val, t_var, omega)
        except Exception as exc:
            raise ValueError(
                f'{where}: Fourier transform of time_expr failed.  '
                f'Either supply freq_image explicitly or simplify '
                f'time_expr to an analytically-tractable form.  '
                f'Original error: {exc}')
        return SR(img)

    def _ft_image(ns, omega):
        out: dict = {}
        param_ns = _ns_var_namespace(ns, [], param_names, [])
        builtins = _builtin_namespace()
        base_ns  = {**builtins, **param_ns, 'omega': omega}
        t_var = SR.var('t')

        for spec in kernel_specs:
            kname = spec['name']
            if not hasattr(ns, kname):
                continue
            ksym_obj = getattr(ns, kname)
            if not spec.get('freq_image') and not spec.get('time_expr'):
                continue
            # New-style ``indexed_by`` (list of populations) wins
            # over legacy ``indexed`` (bool / 'vector' / 'matrix').
            indexed_by = spec.get('indexed_by')
            if indexed_by:
                n_idx = len(indexed_by)
                is_matrix = (n_idx == 2)
                is_vector = (n_idx == 1)
            else:
                indexed = spec.get('indexed', False)
                is_matrix = (indexed == 'matrix')
                is_vector = (indexed is True or indexed == 'vector') \
                    and not is_matrix

            if is_matrix:
                # ``ksym_obj`` is a list-of-lists of SR symbols
                # ``g_<i+1>_<j+1>``.  Evaluate per (i, j) pair with
                # both indices in scope so the expression (freq or
                # time) can reference per-pair parameters like
                # ``tau_g[i, j]``.
                n_rows = len(ksym_obj)
                for i in range(n_rows):
                    row = ksym_obj[i]
                    for j in range(len(row)):
                        eval_ns = {**base_ns, 'i': i, 'j': j}
                        where = (f"kernel {kname} at ({i+1}, {j+1})")
                        img = _compute_image(spec, eval_ns, omega,
                                             t_var, where)
                        if img is not None:
                            out[row[j]] = img
            elif is_vector:
                for i in range(len(ksym_obj)):
                    eval_ns = {**base_ns, 'i': i}
                    where = f'kernel {kname} at {i+1}'
                    img = _compute_image(spec, eval_ns, omega,
                                         t_var, where)
                    if img is not None:
                        out[ksym_obj[i]] = img
            else:
                # Scalar kernel — one substitution.
                where = f"kernel {kname}"
                img = _compute_image(spec, base_ns, omega, t_var, where)
                if img is not None:
                    out[ksym_obj] = img

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

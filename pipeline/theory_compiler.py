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
    heaviside, dirac_delta, I, pi, function as sr_function,
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
                      naming_convention=None) -> dict[str, Any]:
    """Assemble user-name → ns-attribute dict.

    For each declared field/parameter/kernel ``name`` the corresponding
    ``ns.<name>`` is exposed under that name.  When a
    ``naming_convention`` dict is supplied, fields are ALSO exposed
    under their natural names (e.g. ``ns.dn`` accessible as both
    ``dn`` and ``n`` in user expressions).  Matrix-shaped parameters
    are wrapped in :class:`_MatrixView` for tuple-subscript access.
    """
    out: dict[str, Any] = {}

    # Fields (vectors of SR vars).  Expose under internal name AND
    # under natural name when the model declares a translation.
    fluct_natural_to_internal = (
        (naming_convention or {}).get('fluctuation_fields') or {})
    fluct_internal_to_natural = {v: k
                                 for k, v in fluct_natural_to_internal.items()}

    for fname in field_names:
        if not hasattr(ns, fname):
            continue
        val = getattr(ns, fname)
        out[fname] = val
        # Also expose under natural name if there's a mapping
        natural = fluct_internal_to_natural.get(fname)
        if natural and natural != fname:
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
    nsdict.update(_ns_var_namespace(ns, field_names, param_names,
                                    kernel_names,
                                    naming_convention=naming_convention))
    nsdict['pop'] = list(range(n_pop))

    # Compile each user-defined function into a Python callable.
    # Helper functions (non-transfer) always get the concrete binding
    # — they're not Taylor-expanded by the framework.
    for fn in functions:
        if transfer_function and fn['name'] == transfer_function['name']:
            continue   # bound separately below
        nsdict[fn['name']] = _make_function_callable(fn, nsdict)

    # Bind the transfer function based on context.
    if transfer_function is not None:
        tname = transfer_function['name']
        if transfer_function_mode == 'action':
            # User writes ``phi[i](dv[i])`` in the action — the
            # comprehension binds ``i`` to a population index, and
            # ``phi[i]`` returns a formal Sage callable that
            # FieldTheory's auto-Taylor pass expands.
            nsdict[tname] = _IndexedFormalFunction(tname)
        elif transfer_function_mode == 'formal':
            # Legacy: per-i loop in compiler binds i externally.
            if i is not None:
                nsdict[tname] = (lambda x, _i=i, _t=tname:
                                 sr_function(f'{_t}_{_i + 1}')(x))
            else:
                nsdict[tname] = (lambda x, _t=tname:
                                 sr_function(f'{_t}_1')(x))
        else:    # 'concrete'
            nsdict[tname] = _make_function_callable(transfer_function, nsdict)

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

    The action is evaluated as a single Sage expression — the user
    writes the **full** action including any sums over populations.
    Population iteration uses Python comprehension syntax::

        sum(nt[i] * (nstar[i] + dn[i]) - (exp(nt[i]) - 1) * phi[i](dv[i])
            + vt[i] * (...)
            for i in pop)

    Convention: ``pop = range(n_populations)`` is pre-bound.  Terms
    that don't sum over ``pop`` (e.g. external-noise couplings to a
    different population set) are written outside any sum.

    Transfer functions are accessed with **indexed** syntax —
    ``phi[i](dv[i])`` — because the action evaluator no longer has
    an outer ``i`` loop; the index comes from the user's own
    comprehension.  ``phi[i]`` produces a formal Sage function call
    ``function(f'phi_{i+1}')(arg)`` that FieldTheory's auto-Taylor
    pass expands.  Non-transfer functions are bound concretely.
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
        return _safe_eval(action_text, nsdict, 'action')

    return _action


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


def make_specializations_lambda(phi_fn_spec: dict, *, field_names, param_names,
                                kernel_names, functions, n_pop):
    """Auto-derive ``model['specializations']`` from the transfer
    function by Taylor-expanding ``phi(v)`` around ``vstar[i]``.

    Produces the substitutions::

        phi1_<i> = phi'(vstar[i])
        phi2_<i> = phi''(vstar[i])
        ...
        phi<k>_<i> = 0   for k > order(phi)

    that the FieldTheory expander needs.
    """
    expr_text = phi_fn_spec['expression']
    arg_name  = phi_fn_spec['args'][0]

    def _specs(ns):
        base_ns = _build_namespace_for_eval(
            ns,
            field_names = field_names,
            param_names = param_names,
            kernel_names = kernel_names,
            functions   = [],
            n_pop       = n_pop,
        )
        v_sym = SR.var('_phi_taylor_v')
        phi_at_v = _safe_eval(
            expr_text, {**base_ns, arg_name: v_sym},
            f'phi({arg_name}) Taylor expansion')

        out: dict = {}
        order = ns._taylor_order
        for i in range(n_pop):
            for k in range(1, order + 1):
                deriv_at_v = phi_at_v.derivative(v_sym, k)
                deriv_at_saddle = deriv_at_v.subs({v_sym: ns.vstar[i]})
                out[SR.var(f'phi{k}_{i+1}')] = deriv_at_saddle
        return out

    return _specs


def make_mf_bg_conditions_lambda(mf_eqs: dict[str, str],
                                 phi_fn_spec: dict | None,
                                 *,
                                 field_names, param_names, kernel_names,
                                 functions, n_pop,
                                 iteration_saddle: str = 'nstar'):
    """Build ``model['mf_bg_conditions']`` from the user's per-saddle
    equations.

    The transfer function (``phi``) is bound in CONCRETE mode here.
    Auto-emits ``phi0_<i> = ns.nstar[i]`` to close the EOM at the
    saddle for FieldTheory's auto-Taylor pass.

    Convention: ``iteration_saddle`` (default ``'nstar'``) is the
    saddle that ``solve_mean_field`` iterates on numerically.  Its
    own equation (e.g. ``nstar[i] = phi(vstar[i])``) is **not**
    substituted into the action — doing so would break the
    saddle-tadpole cancellation that the framework relies on.  The
    self-consistency ``nstar = phi(vstar)`` is instead encoded
    implicitly via ``phi0_<i> → nstar[i]`` (which is the
    Taylor-coefficient-renaming substitution).  ``solve_mean_field``
    then uses ``phi_concrete`` to evaluate ``phi(vstar)`` numerically
    while iterating ``nstar``.
    """
    def _bg(ns):
        out: dict = {}
        for saddle_name, rhs_text in mf_eqs.items():
            if saddle_name == iteration_saddle:
                # Skip — solver iterates this saddle.  Substituting
                # it would break the (1,0) tadpole cancellation.
                continue
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
                out[saddle_array[i]] = rhs

        # phi0_<i> = nstar[i] — closes the EOM at the saddle.
        if hasattr(ns, 'nstar') and phi_fn_spec is not None:
            for i in range(n_pop):
                out[SR.var(f'phi{0}_{i+1}')] = ns.nstar[i]
        return out

    return _bg


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

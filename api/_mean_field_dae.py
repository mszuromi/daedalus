"""
DAE-based mean-field solver — multi-root + linear-stability ready.

Consumes ``model['equations']`` (declared via
``ModelBuilder.equation(lhs=..., rhs=..., population=...)``) and finds
ALL fixed points of the algebraic system obtained by setting ``Dt → 0``
on every equation's LHS.  Uses multi-start Newton (scipy ``optimize.root``
with the ``hybr`` Powell hybrid method), deduplicates clusters, filters
out non-real solutions, sorts by the first declared physical field's
first population index, and selects ``fixed_point_index``-th root.

Companion ``linear_stability(model, fundamental, root)`` (step c, lands
separately) builds the generalized-eigenvalue problem at a given root
using the same equation parsing infrastructure here.

This module is INTENTIONALLY decoupled from
``api/_mean_field.py`` — the legacy iteration solver still runs
for models without ``.equation(...)`` declarations.  Routing happens
in ``compute_cumulants`` (step d).
"""
from __future__ import annotations

import warnings
from typing import Optional

import math

import numpy as np
from scipy.optimize import root as _scipy_root

from engine.core.dt_order import NonlinearDtError

# Residual value where an equation cannot be evaluated (overflow, log of a
# negative number, ...).  Far above any physical residual in any units (the
# old 1e10 was below the terms of a model with rates ~1e11), and small enough
# that MINPACK's scaled norms stay finite.
_PENALTY = 1e100


# Math functions that may appear in user equation text (LHS / RHS of
# .equation(...), or inside .define_function(...) expressions).
# Drawn from numpy so they work on both scalars and numpy arrays.
def _factorial(x):
    """``x! = Gamma(x + 1)`` for real ``x`` (Sage prints ``gamma(1 - y)`` of
    a symbolic ``y`` as ``factorial(-y)``)."""
    from scipy.special import gamma
    return gamma(np.asarray(x, dtype=float) + 1.0)


def _gamma(x):
    from scipy.special import gamma
    return gamma(np.asarray(x, dtype=float))


_MATH_NS = {
    'tanh': np.tanh, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'exp':  np.exp,  'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
    'sinh': np.sinh, 'cosh': np.cosh,
    'pi':   float(np.pi),
}

# The mean of a library noise (CGMY, tempered stable, ...) with a symbolic
# exponent carries Gamma functions (audit LOUD-02: NameError 'factorial' in
# the saddle solve).  Placed BEFORE the model's names in every namespace: a
# parameter called ``gamma`` keeps its value.
_SPECIAL_NS = {'gamma': _gamma, 'factorial': _factorial}


def _sage_to_python(text: str) -> str:
    """Translate Sage-syntax expression text to plain-Python syntax
    so it parses cleanly under ``eval``.

    The only difference that matters in practice is the power
    operator: Sage and ModelBuilder docstrings use ``^`` (which Sage
    preparses to ``**``), but plain Python's ``^`` is bitwise XOR.
    Without this translation, an expression like ``eps*x[i]^3`` raises
    ``TypeError: ufunc 'bitwise_xor' not supported`` when ``x[i]`` is
    a numpy float.

    Bitwise XOR is never meaningful in MF equations or transfer
    functions over reals, so unconditional replacement is safe.
    """
    return text.replace('^', '**')


# ── Population / field helpers ────────────────────────────────────────


def _pop_size(model: dict, pop_name: Optional[str]) -> int:
    """Return the size of the population with ``pop_name``.  ``None``
    means scalar (return 1)."""
    if pop_name is None:
        return 1
    for p in model.get('populations', []):
        if p['name'] == pop_name:
            return int(p.get('size', 1))
    raise KeyError(
        f"_pop_size: no population named {pop_name!r} in model "
        f"(declared populations: {[p['name'] for p in model.get('populations', [])]})"
    )


def _state_variables(model: dict) -> list[tuple[str, Optional[str], int]]:
    """Return ``[(var_name, pop_name, pop_size), ...]`` in declaration
    order from ``model['physical_fields']``.

    ``var_name`` is the USER-FACING natural name (``'n'``, ``'v'``,
    ...) — what the user types in equation strings.  The internal
    MSR-JD name (``'dn'``, ``'dv'``) is kept on the field dict but is
    irrelevant to the MF solver since equations are eval'd in a Python
    namespace.

    The first entry's `var_name` is the SORT KEY for multi-root
    fixed-point selection — `fixed_point_index=0` returns the root with
    the smallest first-index value of this variable.
    """
    out = []
    for f in model.get('physical_fields', []):
        # Prefer the user-facing natural name; fall back to the
        # internal name for models that didn't declare one.
        var_name = f.get('natural_name') or f['name']
        pop = f.get('population')
        size = _pop_size(model, pop)
        out.append((var_name, pop, size))
    return out


def _state_slices(state_vars: list[tuple[str, Optional[str], int]]
                  ) -> dict[str, tuple[int, int]]:
    """Map ``var_name → (start, end)`` slice into the flat state vector."""
    slices = {}
    offset = 0
    for var, _, size in state_vars:
        slices[var] = (offset, offset + size)
        offset += size
    return slices


# ── Phi/function callable construction ───────────────────────────────


class _PhiCallableList:
    """List of per-population phi callables that ALSO supports bare
    scalar-call (``f(v)``) when there's exactly one population
    position.  Indexed access (``f[i](v)``) always works, mirroring
    the action-side ``_IndexedFormalFunction`` convention.
    """
    __slots__ = ('_callables',)

    def __init__(self, callables):
        self._callables = list(callables)

    def __getitem__(self, i):
        return self._callables[i]

    def __len__(self):
        return len(self._callables)

    def __iter__(self):
        return iter(self._callables)

    def __call__(self, *args, **kwargs):
        # Scalar mode: ``f(v)`` resolves to the single callable.  In
        # multi-pop models the user always writes ``f[i](v)`` so this
        # path is harmless (the test would just go through __getitem__).
        if len(self._callables) != 1:
            raise TypeError(
                f'bare call ``f(...)`` requires exactly one population '
                f'position; got {len(self._callables)}.  Use ``f[i](...)`` '
                f'with an explicit index instead.'
            )
        return self._callables[0](*args, **kwargs)


def _make_phi_callables(model: dict, params: dict) -> dict[str, '_PhiCallableList']:
    """For each declared function (``define_function('phi', ...)``),
    return a ``_PhiCallableList`` — supports both ``f(v)`` (scalar)
    and ``f[i](v)`` (indexed) — that takes a single state-variable
    value and returns the function's value.

    The callable is built by Python ``eval`` on the function's
    expression text with the argument-name bound to the input value
    and ``i`` bound to the population index.

    Returns a dict ``{function_name: _PhiCallableList}``.
    Currently only single-arg functions are supported (the framework's
    main use case for ``phi``).  Multi-arg functions are skipped — the
    DAE solver only needs the saddle value, and indexed multi-arg
    functions used in CGF terms don't appear in equation RHSs.
    """
    out = {}
    for fn_spec in model.get('functions', []):
        # Read text-form spec preserved by ModelBuilder (see
        # ``expression_text`` / ``args_text`` in ``_compile_text_
        # declarations``).  Falls back to the compiled lambda's spec
        # if the user defined the function via the older add_function
        # API (no text available, so we skip).
        args_text = fn_spec.get('args_text') or []
        expr_text = fn_spec.get('expression_text')
        if not args_text or expr_text is None:
            continue
        if len(args_text) != 1:
            continue
        arg_name = args_text[0]
        pop = fn_spec.get('population')
        pop_size_val = _pop_size(model, pop)

        # Translate Sage power ``^`` to Python ``**`` once.
        expr_py = _sage_to_python(expr_text)
        callables = []
        for i in range(pop_size_val):
            def phi_i(x_val, _expr=expr_py, _arg=arg_name,
                      _i=i, _params=params):
                ns = {**_SPECIAL_NS, **_params, **_MATH_NS,
                      _arg: x_val, 'i': _i,
                      'sum': sum, '__builtins__': {}}
                return eval(_expr, ns)
            callables.append(phi_i)
        out[fn_spec['name']] = _PhiCallableList(callables)
    return out


# ── Parameter dict → numpy-ized for eval ─────────────────────────────


def _numpy_params(model: dict, fundamental: dict) -> dict:
    """Convert ``fundamental`` to a dict of numpy arrays so that
    expressions like ``w[i,j]`` (2-D indexing) and ``Em[i]`` (1-D) work
    under plain Python ``eval``.
    """
    from api._mean_field import complete_parameters
    fundamental = complete_parameters(model, fundamental,
                                     'mean-field DAE solve', strict=False)
    out = {}
    for pspec in model.get('parameters', []):
        name = pspec['name']
        if name not in fundamental:
            continue
        val = fundamental[name]
        ib = pspec.get('indexed_by')
        if ib and len(ib) >= 1:
            out[name] = np.asarray(val, dtype=float)
        elif isinstance(val, (list, tuple)):
            out[name] = np.asarray(val, dtype=float)
        else:
            out[name] = float(val)
    return out


# ── Population index set dict ────────────────────────────────────────


def _index_sets(model: dict) -> dict[str, list]:
    """Return ``{pop_name: list(range(size))}`` for every declared
    population.  Used so that ``for j in E`` inside a comprehension
    iterates over the correct range.
    """
    out = {}
    for p in model.get('populations', []):
        out[p['name']] = list(range(int(p.get('size', 1))))
    return out


# ── Residual evaluation ──────────────────────────────────────────────


def _magnitude_text(text: str) -> str:
    """Python text for the sum of the magnitudes of the additive terms of
    ``text`` (``a - b*c`` -> ``abs(a) + abs(b)*abs(c)``; a quotient by the
    magnitude of its divisor's value; any other call or power by the
    magnitude of its value; ``sum(... for ...)`` termwise).  The round-off
    of evaluating ``text`` is ~eps times this, whatever cancels in it."""
    import ast

    def ab(node):
        return ast.Call(func=ast.Name('abs', ast.Load()), args=[node],
                        keywords=[])

    def T(node):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return ast.BinOp(T(node.left), ast.Add(), T(node.right))
            if isinstance(node.op, ast.Mult):
                return ast.BinOp(T(node.left), ast.Mult(), T(node.right))
            if isinstance(node.op, ast.Div):
                return ast.BinOp(T(node.left), ast.Div(), ab(node.right))
            return ab(node)
        if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.USub, ast.UAdd)):
            return T(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'sum' and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)):
            g = node.args[0]
            return ast.Call(func=node.func, args=[ast.GeneratorExp(
                elt=T(g.elt), generators=g.generators)], keywords=[])
        return ab(node)

    try:
        tree = ast.parse(text, mode='eval')
        out = ast.Expression(T(tree.body))
        return ast.unparse(ast.fix_missing_locations(out))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return f'abs({text})'


def _build_residual(model: dict, params_np: dict,
                    phi_callables: dict[str, list],
                    index_sets: dict[str, list],
                    state_vars: list[tuple[str, Optional[str], int]],
                    slices: dict[str, tuple[int, int]]):
    """Return a function ``R(x: np.ndarray) -> np.ndarray`` that
    evaluates the full residual vector at state ``x``.

    Each equation is expanded over its declared population's index
    range.  ``Dt`` is hard-substituted to ``0`` for MF.
    """
    equations = model.get('equations', [])

    # Precompute, per-equation, the expanded (i-bound) (lhs, rhs)
    # source strings to avoid string ops in the hot loop.  Each
    # element of the returned list is (lhs_text, rhs_text, i, pop)
    # describing a SINGLE scalar residual.  Power operator ``^`` is
    # translated to ``**`` here so the strings are valid Python.
    expanded = []
    mag_texts = []          # per row: magnitude texts (``_magnitude_text``)
    for eq in equations:
        pop = eq['population']
        pop_size_val = _pop_size(model, pop) if pop is not None else 1
        lhs_py = _sage_to_python(eq['lhs_text'])
        rhs_py = _sage_to_python(eq['rhs_text'])
        lhs_m, rhs_m = _magnitude_text(lhs_py), _magnitude_text(rhs_py)
        for i in range(pop_size_val):
            expanded.append((lhs_py, rhs_py, i, pop))
            mag_texts.append((lhs_m, rhs_m))

    # Identify single-position state vars so the namespace binds them
    # as bare numpy scalars (size-1 arrays support bare arithmetic
    # via numpy broadcasting; this matches the symbolic stability
    # path that wraps these in ``_FieldScalar``).  Multi-position
    # state vars stay as arrays — ``[i]`` indexing required.
    def R(x: np.ndarray, mag: Optional[np.ndarray] = None) -> np.ndarray:
        # ``mag`` (optional, filled in place): the sum of the magnitudes of
        # each row's terms, the size against which its round-off is judged.
        # Pack state variables as numpy arrays keyed by their declared
        # name (so n[i], v[i] etc. work under Python's eval).
        state = {var: np.asarray(x[start:end], dtype=float)
                 for var, (start, end) in slices.items()}
        # IMPORTANT: pass the namespace as the GLOBALS argument to
        # eval(), not locals.  Python's comprehension-scoping rule
        # (PEP 3104) puts comprehensions in their own function scope
        # that does NOT see the eval-locals dict — but DOES see the
        # eval-globals.  Without this, ``sum(w[i,j]*n[j] for j in E)``
        # blows up with NameError on the free names inside the
        # genexpr.
        ns = {
            **_SPECIAL_NS,
            **params_np,
            **state,
            **phi_callables,
            **index_sets,
            **_MATH_NS,
            'Dt':           0,
            # Spatially-uniform mean field ⇒ ∇²(const) = 0.  Spatial
            # models write a ``Laplacian`` operator in the drift; it
            # must vanish on the homogeneous saddle exactly as ``Dt``
            # does.  Without this the numerical residual eval hits an
            # unbound ``Laplacian`` and the saddle solve goes wrong.
            'Laplacian':    0,
            'sum':          sum,
            '__builtins__': {},
        }
        out = np.empty(len(expanded), dtype=float)
        for k, (lhs_text, rhs_text, i, pop) in enumerate(expanded):
            ns['i'] = i
            try:
                # A saturating expression (``1/(1 + exp(300*x^2))``) overflows
                # to a valid value; numpy's overflow / invalid warning, issued
                # from this frame whose ``__builtins__`` is empty, would fail
                # with ``KeyError('__import__')`` once warnings are shown
                # (pytest, ``-W``).  Silence it; the values are unchanged.
                with np.errstate(all='ignore'):
                    lhs = eval(lhs_text, ns)
                    rhs = eval(rhs_text, ns)
                # A scalar (no-population) model evaluates its state as
                # 1-element arrays; ``float(array)`` is deprecated for
                # ndim > 0 (NumPy 1.25) and will raise, so unwrap explicitly.
                diff = np.asarray(lhs - rhs, dtype=float).reshape(-1)
                if diff.size != 1:
                    raise ValueError(
                        f'residual of equation {k} ({lhs_text!r} = '
                        f'{rhs_text!r}, i={i}) is not a scalar: shape '
                        f'{np.shape(lhs - rhs)}.')
                val = float(diff[0])
                if mag is not None:
                    with np.errstate(all='ignore'):
                        mv = (eval(mag_texts[k][0], ns)
                              + eval(mag_texts[k][1], ns))
                    mv = float(np.asarray(mv, dtype=float).reshape(-1)[0])
                    mag[k] = mv if math.isfinite(mv) else _PENALTY
                # an overflow / invalid operation silenced above gives inf or
                # NaN, never an exception: penalize it like the exceptions
                # below, so a NaN residual can never pass as a root (NaN
                # compares False against every tolerance)
                out[k] = val if math.isfinite(val) else _PENALTY
            except NameError as exc:
                # (still a NameError: the multi-start loop swallows ValueErrors
                # as "no fixed point from this seed")
                raise NameError(
                    f'mean-field DAE solve: {exc} in the equation '
                    f'{lhs_text!r} = {rhs_text!r}.  If it is a model '
                    f'parameter, it was not in the parameters given and '
                    f'declares no default (pass parameters={{...}} or '
                    f'declare default= on ModelBuilder.parameter).') from exc
            except (ValueError, ZeroDivisionError, OverflowError,
                    FloatingPointError):
                # Penalize: send the solver away from this region.
                out[k] = _PENALTY
                if mag is not None:
                    mag[k] = _PENALTY
        return out

    return R, expanded


# ── Initial-guess sampling ───────────────────────────────────────────


def _seed_box_default(state_vars, model, fundamental):
    """Per-variable default sampling box derived from the variable's
    declared ``domain`` and a heuristic scale.

    Heuristic: scale = max(|param value|) over all declared parameters
    (fallback 1.0).  Sample ``[0, 5·scale]`` for a ``'positive'`` OR an
    undeclared domain (the historical default — preserves every existing
    model's seed box); use the SYMMETRIC ``[-3·scale, 3·scale]`` ONLY when a
    field's saddle explicitly declares ``domain='real'``/``'symmetric'`` (a
    double-well opting into the negative half-line).

    Returns ``{var_name: (low, high)}``.
    """
    scale = 0.0
    for pname, pval in fundamental.items():
        arr = np.asarray(pval, dtype=float).ravel()
        if arr.size:
            scale = max(scale, float(np.max(np.abs(arr))))
    if scale <= 0.0:
        scale = 1.0

    # Resolve each variable's domain from its mean-field SADDLE PARAMETER
    # (<natural>star, mean_field=True) — physical fields carry no domain key.
    # The auto-saddle sets 'positive' for rate fields ('n' or an explicit
    # physical_field(domain='positive')), inherits an explicit
    # physical_field(domain='real') for symmetric / double-well order
    # parameters, and is None otherwise.
    saddle_domain = {}
    for p in model.get('parameters', []):
        if p.get('mean_field') and p.get('natural_name'):
            saddle_domain[p['natural_name']] = p.get('domain')
    boxes = {}
    for var, _, _ in state_vars:
        # SYMMETRIC box only for an EXPLICIT 'real'/'symmetric' domain (opt-in
        # for negative wells); 'positive' OR undeclared -> POSITIVE box.  This
        # preserves the historical seed box of every existing model: widening
        # an undeclared field's box (e.g. a voltage) destabilises its
        # multi-start saddle solve and was an env-sensitive regression.
        if saddle_domain.get(var) in ('real', 'symmetric'):
            boxes[var] = (-3.0 * scale, 3.0 * scale)
        else:
            boxes[var] = (0.0, 5.0 * scale)
    return boxes


def _sample_seeds(state_vars, slices, model, fundamental,
                  n_starts, seed_box, rng):
    """Generate ``n_starts`` initial guesses for the state vector.

    Each variable gets sampled uniformly in its box.  ``seed_box``
    overrides defaults per-variable (``{var_name: (low, high)}``).
    """
    boxes = _seed_box_default(state_vars, model, fundamental)
    if seed_box:
        boxes.update(seed_box)
    total = sum(size for _, _, size in state_vars)
    out = np.empty((n_starts, total), dtype=float)
    for var, _, size in state_vars:
        lo, hi = boxes[var]
        start, end = slices[var]
        out[:, start:end] = rng.uniform(lo, hi, size=(n_starts, size))
    return out


#: Decades ``10^j`` (``|j| <= _LADDER_DECADES``) of the unit-free seed ladder
#: and seeds per decade (see ``_ladder_seeds``).
_LADDER_DECADES = 12
_LADDER_PER_DECADE = 3


def _ladder_seeds(state_vars, slices, model, fundamental, seed_box, rng):
    """Initial guesses on boxes of every decade ``10^j``, ``j = -12..12``
    (the shape of each variable's default box: ``[0, 5 s]`` or ``[-3 s,
    3 s]``), ``_LADDER_PER_DECADE`` per decade: the fallback when the
    default box finds no root.  That box scales with the LARGEST parameter
    value, which is not a field scale: in other units (a field in 1e-6
    units, a rate in 1e6 units) the roots sit many decades away from it and
    were missed (G2_NO_SADDLE).  A variable with an explicit ``seed_box``
    keeps its box."""
    unit = _seed_box_default(state_vars, model, {})
    total = sum(size for _, _, size in state_vars)
    blocks = []
    for j in range(-_LADDER_DECADES, _LADDER_DECADES + 1):
        if j == 0:
            continue
        out = np.empty((_LADDER_PER_DECADE, total), dtype=float)
        for var, _, size in state_vars:
            start, end = slices[var]
            if seed_box and var in seed_box:
                lo, hi = seed_box[var]
            else:
                lo, hi = (10.0 ** j) * unit[var][0], (10.0 ** j) * unit[var][1]
            out[:, start:end] = rng.uniform(lo, hi,
                                            size=(_LADDER_PER_DECADE, size))
        blocks.append(out)
    return np.vstack(blocks) if blocks else np.empty((0, total))


#: At most this many root scales get their own second-pass seed box.
_MAX_RESCALE_BOXES = 8
#: Rounds of the second pass (a round boxes the root scales found so far).
_RESCALE_ROUNDS = 3


def _rescale_seeds(roots, state_vars, slices, model, fundamental, seed_box,
                   n_starts, rng, used=None):
    """Second-pass guesses: ``n_starts`` on the default box shape scaled to
    each root found whose size (per variable, its largest component) lies
    below a fifth of the default box (or above it); one box per decade of
    root size.  Variables with an explicit ``seed_box``, and variables at 0
    in that root, keep their default box.  Empty when every root is on the
    default box's scale (the existing models: their seeds are unchanged)."""
    default = _seed_box_default(state_vars, model, fundamental)
    if seed_box:
        default.update(seed_box)
    unit = _seed_box_default(state_vars, model, {})
    keys, boxes = (set() if used is None else used), []
    for r in roots:
        box, key, off = {}, [], False
        for var, _, _ in state_vars:
            a, b = slices[var]
            sv = float(np.max(np.abs(r[a:b]))) if b > a else 0.0
            lo, hi = default[var]
            top = max(abs(lo), abs(hi))
            if sv == 0.0 or (seed_box and var in seed_box) or top == 0.0:
                box[var] = (lo, hi)
                key.append(None)
                continue
            if not (0.2 * top <= 5.0 * sv <= 5.0 * top):
                off = True
            box[var] = (unit[var][0] * sv, unit[var][1] * sv)
            key.append(int(math.floor(math.log10(sv))))
        key = tuple(key)
        if off and key not in keys and len(boxes) < _MAX_RESCALE_BOXES:
            keys.add(key)
            boxes.append(box)
    total = sum(size for _, _, size in state_vars)
    out = []
    for box in boxes:
        blk = np.empty((n_starts, total), dtype=float)
        for var, _, size in state_vars:
            a, b = slices[var]
            blk[:, a:b] = rng.uniform(box[var][0], box[var][1],
                                      size=(n_starts, size))
        out.append(blk)
    return np.vstack(out) if out else np.empty((0, total))


def _undeclared_domain_vars(state_vars, model, seed_box):
    """State variables seeded on the positive box only because their
    saddle declares no domain (and ``seed_box`` does not set them)."""
    declared = {p['natural_name']: p.get('domain')
                for p in model.get('parameters', [])
                if p.get('mean_field') and p.get('natural_name')}
    return [var for var, _, _ in state_vars
            if declared.get(var) is None and var not in (seed_box or {})]


# ── Root refinement, acceptance and deduplication ───────────────────
#
# Every criterion below is relative (to the root, to the row's own linear
# scale, or to the root's own error estimate): rescaling a field or the time
# unit rescales the roots and the residual rows but leaves the verdicts, the
# root set and the selected root unchanged (audit UNIT-02).

#: A converged point is a root when every residual row satisfies
#: ``|R_k| <= _ROOT_RTOL * sum_j |J_kj| w_j`` (``J``: a finite-difference
#: Jacobian at the point; ``w_j = max(|x_j|, 1e-8 c_j)``, ``c_j`` the typical
#: size of variable ``j`` over the converged points): the residual is
#: negligible against the size of the row's linear terms.  (Was an absolute
#: ``max|R| <= 1e-7``: large units were rejected, tiny units accepted points
#: far from the root.)
_ROOT_RTOL = 1e-7
#: The scale floor of a component converging to ZERO, relative to ``c_j``
#: (a zero has no scale of its own).
_ZERO_FLOOR = 1e-8
#: Chord-Newton refinement steps after the hybrid solve (hybr stops at a
#: relative step ~1.5e-8, leaving saddles accurate to ~1e-11 only: ACC-21).
_REFINE_STEPS = 8
#: Linear stability: ``Re sigma < -_EIG_RTOL * r`` for every mode, ``r`` the
#: model's fastest rate (``_rate_scale``).
_EIG_RTOL = 1e-12


def _rate_scale(A, B, sigmas=()):
    """The fastest rate of the pencil ``sigma A + B``: over the rows with a
    ``Dt`` term, ``|B_kj| / |A_kj|`` at the row's largest ``A`` entry ``j``
    (one column: the field units cancel; time units multiply it).  Falls
    back to ``max |sigma|`` when every such ratio is 0 (a pure-integrator
    pencil), and to 0."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = 0.0
    for a, b in zip(A, B):
        if a.size and np.max(np.abs(a)) > 0:
            j = int(np.argmax(np.abs(a)))
            out = max(out, abs(float(b[j])) / abs(float(a[j])))
    if out == 0.0 and len(sigmas):
        out = float(np.max(np.abs(sigmas)))
    return out


def _hybr_point(R, x0):
    """hybr from ``x0``: the converged point (components far below the seed
    snapped to an exact 0 when that does not increase any residual row), or
    ``None``."""
    try:
        sol = _scipy_root(R, x0, method='hybr')
    except (ValueError, ZeroDivisionError, OverflowError,
            FloatingPointError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return None
    x = np.asarray(sol.x, dtype=float)
    f = R(x)
    if not np.all(np.isfinite(f)):
        return None
    return _snap_zeros(R, x, f, (x != 0) & (np.abs(x) <= 1e-8 * np.abs(x0)))[0]


def _snap_zeros(R, x, f, mask):
    """``(x, f)`` with the components ``mask`` set to an exact 0 when that
    does not increase any residual row beyond its round-off (a root at 0
    converges to values like 1e-20 that no relative test can tell from 0),
    else unchanged."""
    if not np.any(mask):
        return x, f
    xs = np.where(mask, 0.0, x)
    m = np.empty(len(f))
    fs = R(xs, m)
    noise = 16.0 * np.finfo(float).eps * m
    if np.all(np.isfinite(fs)) and np.all(np.abs(fs) <= np.abs(f) + noise):
        return xs, fs
    return x, f


def _typical_scale(points):
    """Per component: the median of the nonzero ``|x_j|`` over ``points``
    (0 where every point is 0 there)."""
    P = np.abs(np.asarray(points, dtype=float))
    out = np.zeros(P.shape[1])
    for j in range(P.shape[1]):
        nz = P[:, j][P[:, j] > 0]
        if nz.size:
            out[j] = float(np.median(nz))
    return out


#: Largest growth (decades) of a finite-difference step over a residual that
#: is flat in floating point (see ``_fd_jacobian``).
_FD_MAX_DECADES = 10


def _fd_jacobian(R, x, f, c):
    """``(J, res)``: forward-difference Jacobian at ``x`` (``f = R(x)``),
    steps ``1e-7 max(|x_j|, c_j)`` (relative: no unit enters), or ``None``
    when a column cannot be formed.

    A column over which the residual changes by round-off only (16 ulps of
    the row's terms ``|lhs| + |rhs|``: an equation the builder wrote as
    ``x = x + f(x)`` cancels to round-off wherever ``f(x)`` is below the
    round-off of ``x``, around a degenerate root or far out where a drift
    decays) is retried with the step grown by decades, both
    ways, up to ``_FD_MAX_DECADES``; ``res_j`` is the step that resolved it
    (or the largest tried), 0 when the first step did -- the distance over
    which the point is not resolved, i.e. an error estimate."""
    n = len(x)
    J = np.empty((len(f), n))
    res = np.zeros(n)
    m = np.empty(len(f))
    R(x, m)
    # a change below 16 ulps of the row's terms is round-off, not a slope
    noise = 16.0 * np.finfo(float).eps * m

    def ok(v):
        return np.all(np.isfinite(v)) and np.max(np.abs(v)) < _PENALTY

    for j in range(n):
        h0 = 1e-7 * max(abs(x[j]), c[j])
        if h0 == 0.0:
            # every point is 0 in this component: probe at the largest
            # typical scale of the others (only the column's linear part
            # matters, and R is evaluated in floating point)
            h0 = 1e-7 * (float(np.max(c)) if np.max(c) > 0 else 1.0)
        col = None
        for k in range(_FD_MAX_DECADES + 1):
            h = h0 * 10.0 ** k
            for hh in (h, -h):
                e = np.zeros(n)
                e[j] = hh
                fp = R(x + e)
                if not ok(fp):
                    continue
                if col is None:
                    col = (fp - f) / hh          # first finite column
                if np.any(np.abs(fp - f) > noise):
                    col = (fp - f) / hh
                    break
            else:
                if col is None and k == 0:
                    return None                 # no finite neighbour at all
                continue
            break
        if col is None:
            return None
        res[j] = 0.0 if k == 0 else h
        J[:, j] = col
        # the component is determined only to the round-off of the rows it
        # enters (a value below that is indistinguishable from 0)
        nz = np.abs(col) > 0
        if np.any(nz):
            res[j] = max(res[j], float(np.min(noise[nz] / np.abs(col[nz]))))
    return J, res


def _refine_root(R, x, c):
    """Refine a converged point and decide whether it is a root.

    Chord-Newton steps with a finite-difference Jacobian (kept while the
    relative residual decreases), a snap of components below their own error
    estimate to an exact 0 (``_snap_zeros``), then the relative acceptance
    test (``_ROOT_RTOL``).  Returns ``(x, err)`` (``err``: componentwise
    error estimate, the size of the last Newton correction) or ``None``
    when the point is not a root."""
    x = np.asarray(x, dtype=float)
    f = R(x)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(f))):
        return None
    fd = _fd_jacobian(R, x, f, c)
    if fd is None or not np.all(np.isfinite(fd[0])):
        return None
    J, res = fd
    absJ = np.abs(J)

    def rel(f, x):
        rho = absJ @ np.maximum(np.abs(x), _ZERO_FLOOR * c)
        return _rel_resid(f, rho), rho

    q, rho = rel(f, x)
    err = np.zeros(len(x))
    for _ in range(_REFINE_STEPS):
        if q == 0.0:
            break
        try:
            d = np.linalg.lstsq(J, f, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            break
        if not np.all(np.isfinite(d)):
            break
        xn = x - d
        fn = R(xn)
        if not (np.all(np.isfinite(xn)) and np.all(np.isfinite(fn))):
            break
        qn, rn = rel(fn, xn)
        err = np.abs(d)
        if not qn < q:
            break
        x, f, q, rho = xn, fn, qn, rn
    err = np.maximum(err, res)
    # each component whose value changes no row beyond its round-off (to
    # first order and in fact) is set to an exact 0: it is 0 within what
    # the equations resolve
    snapped = False
    m = np.empty(len(f))
    R(x, m)
    noise = 16.0 * np.finfo(float).eps * m
    for j in np.flatnonzero(x != 0):
        # the value's own contribution must be round-off (else 0 being
        # another root would pass the residual test below)
        if np.any(np.abs(J[:, j] * x[j]) > noise):
            continue
        mask = np.zeros(len(x), dtype=bool)
        mask[j] = True
        xs, fs = _snap_zeros(R, x, f, mask)
        if xs is not x:
            err[j] = max(err[j], abs(x[j]))
            x, f, snapped = xs, fs, True
    if snapped:
        q, rho = rel(f, x)
    if not np.all((f == 0) | (np.abs(f) <= _ROOT_RTOL * rho)):
        return None
    if np.any((x != 0) & (err >= np.abs(x))):
        return None                    # its position is not resolved
    return x, err


def _rel_resid(f, rho):
    """``max_k |f_k| / rho_k`` (``0/0`` counts as 0, ``f/0`` as inf)."""
    af = np.abs(f)
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(af == 0, 0.0, af / rho)
    return float(np.max(q)) if q.size else 0.0


def _roots_from_points(R, points, rtol, atol):
    """Cluster converged hybr points (agreeing to ``rtol`` relative per
    component), refine one representative per cluster (``_refine_root``),
    and return ``(roots, errors)`` of the accepted ones, deduplicated."""
    if not points:
        return [], []
    c = _typical_scale(points)
    reps = _dedup_roots(points, rtol=rtol, atol=atol)
    roots, errs = [], []
    for x in reps:
        hit = _refine_root(R, x, c)
        if hit is not None:
            roots.append(hit[0])
            errs.append(hit[1])
    return _dedup_roots(roots, rtol=rtol, atol=atol, errs=errs)


def _dedup_roots(roots: list, rtol=1e-6, atol=0.0, errs=None) -> list:
    """Cluster roots, keep one representative each (discovery order; the
    caller sorts).  Two roots are one when every component agrees to
    ``rtol`` relative, ``atol`` absolute (0 by default: an absolute floor
    merged distinct roots of a field in small units), or within ten times
    their error estimates ``errs`` (two approximations of the same root).
    With ``errs`` given, returns ``(roots, errs)``."""
    out, out_e = [], []
    if errs is None:
        errs_in = [np.zeros_like(np.asarray(r, dtype=float)) for r in roots]
    else:
        errs_in = errs
    for r, e in zip(roots, errs_in):
        merged = False
        for q, (kept, ke) in enumerate(zip(out, out_e)):
            tol = (atol + rtol * np.maximum(np.abs(r), np.abs(kept))
                   + 10.0 * (e + ke))
            if np.all(np.abs(r - kept) <= tol):
                merged = True
                # keep the better-resolved representative
                if float(np.max(e, initial=0.0)) < float(np.max(ke, initial=0.0)):
                    out[q], out_e[q] = r, e
                break
        if not merged:
            out.append(r)
            out_e.append(e)
    return out if errs is None else (out, out_e)


# ── Top-level solver ─────────────────────────────────────────────────


def _saddle_keys(model, state_vars) -> dict:
    """``{state variable: saddle parameter name}`` for the ``.equation``
    form (its state variables are the physical fields): the field's
    declared saddle (``api.naming.saddle_for_field``), else
    ``<field>star``, the name ``physical_field`` gives it."""
    from api.naming import saddle_for_field
    return {var: (saddle_for_field(model, var) or f'{var}star')
            for var, _, _ in state_vars}


def _param_subs(model, fundamental, pop_size_map) -> dict:
    """``{parameter SR symbol: value}`` in the namespace convention (a
    vector parameter ``p`` -> ``p1, p2, ...``, a matrix -> ``p11, ...``)."""
    from sage.all import SR
    out: dict = {}
    for pspec in model.get('parameters', []):
        pname = pspec['name']
        if pname not in fundamental:
            continue
        val = fundamental[pname]
        ib = pspec.get('indexed_by')
        if ib and len(ib) == 2:
            for i in range(pop_size_map.get(ib[0], 1)):
                for j in range(pop_size_map.get(ib[1], 1)):
                    out[SR.var(f'{pname}{i+1}{j+1}')] = float(val[i][j])
        elif ib and len(ib) == 1:
            for i in range(pop_size_map.get(ib[0], 1)):
                out[SR.var(f'{pname}{i+1}')] = float(val[i])
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                out[SR.var(f'{pname}{i+1}')] = float(v)
        else:
            out[SR.var(pname)] = float(val)
    return out


def _pop_sizes(model, ns=None) -> dict:
    sizes = dict(getattr(ns, '_pop_size', None) or {})
    if not sizes:
        sizes = {p['name']: int(p.get('size', 1))
                 for p in model.get('populations') or []}
    return sizes


def _saddle_arrays(model, ns) -> dict:
    """``{saddle name: [SR symbol per population index]}`` for the declared
    saddles (``mean_field=True``) the namespace carries."""
    from api.naming import saddle_names
    out = {}
    for name in saddle_names(model):
        arr = getattr(ns, name, None)
        if arr is None:
            continue
        out[name] = list(arr) if isinstance(arr, (list, tuple)) else [arr]
    return out


def _kernel_symbols(model, ns) -> list:
    out = []
    for kspec in model.get('kernels', []):
        kval = getattr(ns, kspec['name'], None)
        if kval is None:
            continue
        stack = [kval]
        while stack:
            v = stack.pop()
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                out.append(v)
    return out


def _compile_float(expr, syms):
    """A float function of ``syms`` evaluating the SR ``expr``
    (``fast_callable``; plain substitution when it cannot compile)."""
    from sage.all import fast_callable
    try:
        return fast_callable(expr, vars=syms, domain=float)
    except (TypeError, ValueError, NotImplementedError, AttributeError,
            KeyError):
        pass

    def g(*vals, _e=expr, _s=tuple(syms)):
        return float(_e.subs(dict(zip(_s, vals))))
    return g


def _saddle_system(model, fundamental, ns, reduced=False):
    """The saddle equations ``s - rhs_s(saddles) = 0`` of a model declared
    with ``.set_mf_equation`` (``model['mf_bg_conditions'](ns)``: one RHS
    per saddle element, the parameters and each kernel at its integral
    substituted).  Returns ``(state_vars, slices, R, rows, keys)`` in the
    shape of the ``.equation`` system: one state variable per saddle with
    an equation, labelled by its field's natural name (that is what
    ``seed_box`` and the domain lookups use) or by its own name, physical
    fields' saddles first in field order (the root sort key).

    ``reduced=True``: the unknowns are the iteration (closure) saddles only
    (``model['iteration_saddles']``), every other saddle substituted by its
    equation (as the former iteration solver did); the fallback when the
    full system has no root on the default search (a compound saddle far on
    the negative axis, e.g. a voltage of -100, outside the seed box).
    Returns a sixth item ``extras(x) -> {compound saddle: [values]}``
    (``None`` for the full system), or ``None`` altogether when the model
    cannot be reduced."""
    from sage.all import SR
    from sage.symbolic.operators import add_vararg
    from api._mean_field import _kernel_saddle_subs, complete_parameters
    from api.naming import field_for_saddle

    fundamental = complete_parameters(model, fundamental,
                                     'mean-field solve', strict=False)
    mf_bg = model.get('mf_bg_conditions', lambda ns: {})(ns)
    arrays = _saddle_arrays(model, ns)
    unknown = {n: a for n, a in arrays.items() if all(s in mf_bg for s in a)}
    if not unknown:
        raise ValueError(
            'mean-field solve: the model declares no mean-field equation '
            '(no .equation(...) rows and no .set_mf_equation(...) for a '
            'saddle parameter).')
    compound = {}
    if reduced:
        iters = [n for n in (model.get('iteration_saddles') or [])
                 if n in unknown]
        if not iters or len(iters) == len(unknown):
            return None
        compound = {n: unknown.pop(n) for n in list(unknown) if n not in iters}
    field_order = [f.get('natural_name') or f['name']
                   for f in model.get('physical_fields') or []]

    def rank(name):
        nat = field_for_saddle(model, name)
        return (field_order.index(nat) if nat in field_order
                else len(field_order))
    names = sorted(unknown, key=rank)        # stable: declaration order next
    pspec = {p['name']: p for p in model.get('parameters', [])}
    populations = model.get('populations') or []
    state_vars, keys = [], {}
    for name in names:
        label = field_for_saddle(model, name) or name
        ib = pspec[name].get('indexed_by') or []
        pop = ib[0] if ib else (populations[0]['name'] if len(populations) == 1
                                else None)
        state_vars.append((label, pop, len(unknown[name])))
        keys[label] = name
    slices = _state_slices(state_vars)
    syms = [s for name in names for s in unknown[name]]

    psubs = _param_subs(model, fundamental, _pop_sizes(model, ns))
    csyms = [c for arr in compound.values() for c in arr]
    ksubs = _kernel_saddle_subs(ns, model, _kernel_symbols(model, ns),
                                [mf_bg[s] for s in syms + csyms])
    # compound saddles as functions of the unknowns (substituted until none
    # is left; a cycle among them means the model cannot be reduced)
    csub = {c: SR(mf_bg[c]).subs(ksubs).subs(psubs) for c in csyms}
    for _ in range(len(csyms) + 1):
        if not any(set(e.variables()) & set(csyms) for e in csub.values()):
            break
        csub = {c: e.subs(csub) for c, e in csub.items()}
    else:
        return None
    rows, terms, labels = [], [], []
    for name in names:
        for i, s in enumerate(unknown[name]):
            rhs = SR(mf_bg[s]).subs(ksubs).subs(psubs)
            if csub:
                rhs = rhs.subs(csub)
            free = sorted(str(v) for v in set(rhs.variables()) - set(syms))
            if free:
                raise NameError(
                    f'mean-field solve: the saddle equation of {name}[{i}] '
                    f'({s} = {mf_bg[s]}) uses {", ".join(free)}, which have '
                    f'no value.  If they are model parameters, they were not '
                    f'in the parameters given and declare no default (pass '
                    f'parameters={{...}} or declare default= on '
                    f'ModelBuilder.parameter).')
            # the additive terms of the RHS: with the saddle itself, the
            # sizes a row's round-off is judged against
            ops = rhs.operands() if rhs.operator() is add_vararg else [rhs]
            rows.append(_compile_float(s - rhs, syms))
            terms.append([_compile_float(SR(t), syms) for t in ops])
            labels.append((name, i))
    n = len(rows)

    def R(x, mag=None):
        vals = [float(v) for v in x]
        out = np.empty(n, dtype=float)
        for k in range(n):
            try:
                with np.errstate(all='ignore'):
                    v = float(rows[k](*vals))
                out[k] = v if math.isfinite(v) else _PENALTY
            except (ValueError, ZeroDivisionError, OverflowError,
                    FloatingPointError, ArithmeticError, TypeError):
                out[k] = _PENALTY
            if mag is not None:
                # row k is the equation of unknown k: |s_k| + sum |terms|
                try:
                    with np.errstate(all='ignore'):
                        mv = abs(vals[k]) + sum(abs(float(t(*vals)))
                                                for t in terms[k])
                    mag[k] = mv if math.isfinite(mv) else _PENALTY
                except (ValueError, ZeroDivisionError, OverflowError,
                        FloatingPointError, ArithmeticError, TypeError):
                    mag[k] = _PENALTY
        return out

    extras = None
    if compound:
        cfun = {name: [_compile_float(csub[c], syms) for c in arr]
                for name, arr in compound.items()}

        def extras(x):
            vals = [float(v) for v in x]
            return {name: [float(f(*vals)) for f in fs]
                    for name, fs in cfun.items()}
    return state_vars, slices, R, labels, keys, extras


def solve_mean_field_dae(
        model: dict,
        fundamental: dict,
        *,
        n_starts: int = 64,
        fixed_point_index: int = 0,
        seed_box: Optional[dict[str, tuple[float, float]]] = None,
        rtol: float = 1e-6,
        atol: float = 0.0,
        verbose: bool = False,
        rng_seed: int = 0,
        ns=None,
        _reduced: Optional[bool] = None,
) -> dict:
    """Solve the mean-field system of a model: the ``Dt -> 0`` limit of its
    ``ModelBuilder.equation(...)`` rows, or, for a model declared with
    ``.set_mf_equation(saddle, rhs)`` instead, the saddle equations
    ``saddle - rhs = 0`` (read from ``model['mf_bg_conditions'](ns)``; needs
    the FieldTheory namespace ``ns``).  Every model gets the same multi-root
    search (audit GEN-02: the second form used a separate single-root
    iteration from four fixed unit-scale guesses).

    Parameters
    ----------
    model : dict
        Model dict from ``ModelBuilder.build()``.  Must contain
        ``model['equations']`` (non-empty) — equation residuals
        ``LHS - RHS = 0``, ``Dt`` allowed on LHS only.
    fundamental : dict
        Concrete parameter values keyed by parameter name.
    n_starts : int, default 64
        Number of random initial guesses for multi-start Newton on the
        default box.  Then ``n_starts`` more on a box at the scale of each
        root found far off the default box's scale (``_rescale_seeds``:
        the box follows the field's units), and, when the default box finds
        no root at all, a ladder of ``3 x 24`` guesses on boxes of every
        decade ``10^-12 .. 10^12`` first (``_ladder_seeds``).
    fixed_point_index : int, default 0
        Which sorted root to pick as the primary MF.  Sorted ascending
        by the first declared physical field's first population index.
        Out-of-range values are clamped with a warning.
    seed_box : dict, optional
        Per-variable initial-guess box override: ``{'v': (low, high)}``.
        Variables not listed fall back to a domain-aware default
        (``positive`` → ``[0, 5·scale]``; anything else → ``[-3·scale,
        3·scale]``).
    rtol, atol : float
        Tolerances for clustering converged roots (componentwise:
        ``rtol`` relative, ``atol`` absolute, default 0; two approximations
        of one root also merge within their error estimates).
    verbose : bool
        Print a one-line summary per seed (for debugging).
    rng_seed : int
        Seed for the initial-guess RNG (kept reproducible).

    Returns
    -------
    dict with keys:
        ``'mf_values'`` — selected root as ``{<saddle>: [vals]}`` (the
            saddle parameter of each state variable, ``api.naming``).
        ``'mf_all_roots'`` — full sorted list of distinct roots, each
            a dict shaped like ``mf_values``.
        ``'mf_index_used'`` — int, actual index used (after clamping).
        ``'state_var_order'`` — list of state variable names in
            declaration order (i.e. the sort-key origin).
        ``'n_seeds_converged'`` — diagnostic: how many seeds led to a
            valid root before dedup.
    """
    if model.get('equations'):
        state_vars = _state_variables(model)
        if not state_vars:
            raise ValueError(
                "solve_mean_field_dae: no physical fields declared — "
                "there's nothing to solve for."
            )
        slices = _state_slices(state_vars)
        params_np = _numpy_params(model, fundamental)
        phi_callables = _make_phi_callables(model, params_np)
        idx_sets = _index_sets(model)
        R, expanded = _build_residual(
            model, params_np, phi_callables, idx_sets, state_vars, slices)
        keys = _saddle_keys(model, state_vars)
        extras = None
    else:
        if ns is None:
            raise ValueError(
                "solve_mean_field_dae: the model declares no "
                ".equation(...) rows; its .set_mf_equation(...) saddle "
                "equations need the FieldTheory namespace (ns=).")
        # The iteration (closure) saddles are the unknowns when the others
        # can be substituted -- the former iteration solver's system: a
        # compound saddle (a voltage, often negative, with no declared
        # domain) then neither enters the seed box nor the root order, and a
        # model written with or without its voltage selects the same root
        # (audit W6).  The full system otherwise, or when this finds none.
        system = None
        if _reduced is not False:
            system = _saddle_system(model, fundamental, ns, reduced=True)
        used_reduced = system is not None
        if system is None:
            system = _saddle_system(model, fundamental, ns, reduced=False)
        state_vars, slices, R, expanded, keys, extras = system
    total = sum(size for _, _, size in state_vars)

    # Sanity: number of residuals must match number of unknowns.
    if len(expanded) != total:
        raise ValueError(
            f"solve_mean_field_dae: {len(expanded)} equations "
            f"(after population expansion) but {total} state-variable "
            f"unknowns.  The system is "
            f"{'under' if len(expanded) < total else 'over'}-determined."
        )

    rng = np.random.default_rng(rng_seed)
    seeds = _sample_seeds(
        state_vars, slices, model, fundamental, n_starts, seed_box, rng)
    undeclared = _undeclared_domain_vars(state_vars, model, seed_box)
    mirror_cols = (np.concatenate([np.arange(*slices[v]) for v in undeclared])
                   if undeclared else None)

    def mirrored(block):
        m = np.array(block, dtype=float, copy=True)
        m[:, mirror_cols] = -m[:, mirror_cols]
        return m

    points: list[np.ndarray] = []          # from the seeds: selectable
    mirror_points: list[np.ndarray] = []   # from the mirrored seeds

    def run(block, out):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for s, x0 in enumerate(block):
                x = _hybr_point(R, x0)
                if x is None:
                    continue
                out.append(x)
                if verbose:
                    print(f"  seed {s:3d}: converged to {x}")

    run(seeds, points)
    deduped, root_errs = _roots_from_points(R, points, rtol, atol)
    if not deduped:
        # nothing on the default box (it scales with the largest parameter,
        # not with the fields): the unit-free decade ladder
        ladder = _ladder_seeds(state_vars, slices, model, fundamental,
                               seed_box, rng)
        run(ladder, points)
        deduped, root_errs = _roots_from_points(R, points, rtol, atol)
        seeds = np.vstack([seeds, ladder])
    # second pass: a box at the scale of every root found off the default
    # box's scale (the decade ladder has few seeds per decade, so a narrow
    # basin, e.g. the middle root of a cubic, can be missed there)
    # ... repeated for the root scales a pass discovers (a box at one
    # root's scale can reach a root of another scale whose basin the first
    # boxes miss; audit W6: an exactly rescaled cubic lost its largest root),
    # each scale boxed once, at most _RESCALE_ROUNDS rounds
    n_mirror = len(seeds)
    used = set()
    for _ in range(_RESCALE_ROUNDS):
        extra = _rescale_seeds(deduped, state_vars, slices, model,
                               fundamental, seed_box, n_starts, rng, used)
        if not len(extra):
            break
        n_before = len(deduped)
        run(extra, points)
        deduped, root_errs = _roots_from_points(R, points, rtol, atol)
        seeds = np.vstack([seeds, extra])
        if len(deduped) == n_before:
            break
    n_converged = len(points)
    unsearched = []
    mirror_only, mirror_only_e = [], []
    if mirror_cols is not None:
        # The default seeds (and the ladder, when it ran) mirrored in the
        # fields that declare no domain; the roots they reach are selectable
        # like any other.  (They were only reported: a negative root was
        # selectable when a seed happened to overshoot to it, which depends
        # on the field's units -- an exactly rescaled cubic lost a root;
        # audit W6.)  Not the second-pass boxes: their roots are reached
        # from seeds that are mirrored already.
        run(mirrored(seeds[:n_mirror]), mirror_points)
        cand, cand_e = _roots_from_points(R, mirror_points, rtol, atol)
        for x, e in zip(cand, cand_e):
            known = list(deduped) + mirror_only
            if len(_dedup_roots(known + [x], rtol=rtol, atol=atol,
                                errs=list(root_errs) + mirror_only_e + [e])[0]
                   ) > len(known):
                mirror_only.append(x)
                mirror_only_e.append(e)

    # Sort ascending by first declared physical field's first index.
    first_var = state_vars[0][0]
    first_start = slices[first_var][0]
    order = sorted(range(len(deduped)),
                   key=lambda q: float(deduped[q][first_start]))
    deduped = [deduped[q] for q in order]
    root_errs = [root_errs[q] for q in order]
    # the roots only the mirrored seeds reach follow, ascending (where the
    # linearization orders roots by stability, below, all are ordered
    # together: then the order does not depend on which search found a root)
    mo = sorted(range(len(mirror_only)),
                key=lambda q: float(mirror_only[q][first_start]))
    deduped = deduped + [mirror_only[q] for q in mo]
    root_errs = list(root_errs) + [mirror_only_e[q] for q in mo]

    if not deduped and not model.get('equations') and used_reduced:
        # nothing for the iteration saddles alone: the full system
        return solve_mean_field_dae(
            model, fundamental, n_starts=n_starts,
            fixed_point_index=fixed_point_index, seed_box=seed_box,
            rtol=rtol, atol=atol, verbose=verbose, rng_seed=rng_seed, ns=ns,
            _reduced=False)
    if not deduped:
        raise ValueError(
            f"solve_mean_field_dae: no MF fixed point found from "
            f"{n_starts} seeds.  Try increasing n_starts, passing "
            f"explicit seed_box, or checking the equations for typos."
        )

    def _build_root_dict(x, err=False):
        out = {keys[var]: x[slices[var][0]:slices[var][1]].tolist()
               for var, _, _ in state_vars}
        if extras is not None:
            ex = extras(x)
            out.update({k: [0.0] * len(v) for k, v in ex.items()} if err
                       else ex)
        return out

    # Stability classification is gated by ``model['stability_analysis']``
    # (default: False).  Models whose equations are ALL algebraic
    # (typical when voltages have been integrated out) have no
    # differential structure — the generalized-eigenvalue ``(σA + B)``
    # has A ≡ 0, every eigenvalue lands at infinity, and "linear
    # stability" has no physical meaning.  Such models should leave
    # the toggle OFF; ``fixed_point_index`` then ranges over ALL
    # converged roots, sorted ascending by the first state variable.
    #
    # When the toggle is ON: classify every root, filter to the stable
    # subset, sort that subset, expose ``fixed_point_index`` over it
    # (and raise if NO stable root exists — the diagrammatic expansion
    # is undefined at any saddle the DAE solver picks otherwise).
    stab_on = bool(model.get('stability_analysis', False))
    if stab_on and not model.get('equations'):
        raise ValueError(
            "solve_mean_field_dae: stability_analysis(True) classifies the "
            "roots of the model's .equation(...) rows (their Dt terms); this "
            "model declares its saddles with .set_mf_equation(...) only.  "
            "Declare the dynamics with .equation(lhs=..., rhs=...), or turn "
            "stability_analysis off.")

    all_root_records = []
    rec_errs = {}           # id(record) -> its error estimates
    if stab_on and mirror_only:
        # the stable subset is ordered ascending over every root, whichever
        # search reached it (the order does not depend on the units)
        asc = sorted(range(len(deduped)),
                     key=lambda q: float(deduped[q][first_start]))
        deduped = [deduped[q] for q in asc]
        root_errs = [root_errs[q] for q in asc]
    if stab_on:
        for x, xe in zip(deduped, root_errs):
            rd = _build_root_dict(x)
            try:
                stab = linear_stability(model, fundamental, rd, verbose=False)
            except NonlinearDtError:
                # A model nonlinear in Dt is not a flaky-stability
                # nuisance — it is unrepresentable.  Let the actionable
                # message out instead of demoting the root to "unknown".
                raise
            except Exception as e:
                # Defensive: if stability fails for some reason, mark as
                # unknown and skip rather than blowing up the solve.
                stab = {
                    'stable': False,
                    'eigenvalues_finite': np.array([], dtype=complex),
                    'eigenvalues_all': np.array([], dtype=complex),
                    'A': None, 'B': None,
                    'unstable_eigenvalues': [],
                    'error': f'{type(e).__name__}: {e}',
                }
            all_root_records.append({
                'values':              rd,
                'stable':              bool(stab.get('stable', False)),
                'eigenvalues_finite':  np.asarray(stab.get('eigenvalues_finite', [])),
            })
            rec_errs[id(all_root_records[-1])] = _build_root_dict(xe, err=True)
    else:
        # No stability filter.  Tag each record with ``stable=None`` so
        # callers can distinguish "not classified" from "classified
        # unstable".  Where the linearization decides (``.equation`` rows
        # with Dt terms, no kernels), the linearly stable roots come first
        # (each group nearest to 0 first): the default fixed_point_index = 0 is then a
        # stable root whenever one exists, whatever the roots' order on the
        # axis (audit W6: the units decided which root was found, and so
        # which one index 0 was).
        stable_first = set()
        if model.get('equations') and not model.get('kernels') and len(deduped) > 1:
            for q, x in enumerate(deduped):
                try:
                    if linear_stability(model, fundamental, _build_root_dict(x),
                                        verbose=False)['stable']:
                        stable_first.add(q)
                except NonlinearDtError:
                    raise
                except Exception:                        # noqa: BLE001
                    pass
        if stable_first:
            # each group nearest to 0 first (ties ascending): invariant under
            # a change of the field's units, and the root the old search
            # (seeded on x >= 0 from 0) reached first stays first
            import functools

            def _cmp(p, q):
                a = float(deduped[p][first_start])
                b = float(deduped[q][first_start])
                if abs(abs(a) - abs(b)) > 1e-9 * max(abs(a), abs(b)):
                    return -1 if abs(a) < abs(b) else 1
                return (a > b) - (a < b)        # a tie (+-c): ascending
            asc = sorted(range(len(deduped)), key=functools.cmp_to_key(_cmp))
            order = ([q for q in asc if q in stable_first]
                     + [q for q in asc if q not in stable_first])
        else:
            order = list(range(len(deduped)))
        for q in order:
            x, xe = deduped[q], root_errs[q]
            all_root_records.append({
                'values':              _build_root_dict(x),
                'stable':              None,
                'eigenvalues_finite':  np.array([], dtype=complex),
            })
            rec_errs[id(all_root_records[-1])] = _build_root_dict(xe, err=True)

    if stab_on:
        # Pre-filter to stable subset.
        selectable_records = [r for r in all_root_records if r['stable']]
        if not selectable_records:
            raise ValueError(
                f"solve_mean_field_dae: found {len(all_root_records)} "
                f"fixed point(s) but NONE were classified linearly "
                f"stable.  Every root has at least one finite eigenvalue "
                f"with non-negative real part — the diagrammatic "
                f"expansion is undefined at all of them.  Inspect the "
                f"eigenvalues via ``linear_stability(model, fundamental, "
                f"root)`` and revisit the equations or parameters.\n"
                f"(If your equations are all algebraic — voltages "
                f"integrated out, no Dt anywhere — set "
                f"``.stability_analysis(False)`` on the ModelBuilder; "
                f"``fixed_point_index`` will then range over every "
                f"converged root.)"
            )
        index_label = 'stable root(s)'
    else:
        # All converged roots are selectable.
        selectable_records = all_root_records
        index_label = 'converged root(s)'

    n_selectable = len(selectable_records)
    if fixed_point_index < 0 or fixed_point_index >= n_selectable:
        clamped = max(0, min(fixed_point_index, n_selectable - 1))
        warnings.warn(
            f"solve_mean_field_dae: requested fixed_point_index="
            f"{fixed_point_index} but only {n_selectable} "
            f"{index_label} found (out of {len(all_root_records)} "
            f"total); using index {clamped}.",
            stacklevel=2,
        )
        fixed_point_index = clamped

    selected = selectable_records[fixed_point_index]['values']
    selected_errs = rec_errs[id(selectable_records[fixed_point_index])]
    pure = _pure_integrator_rows(model, expanded)
    if pure:
        x_sel = np.concatenate([np.asarray(selected[keys[v]], dtype=float)
                                for v, _, _ in state_vars])
        # finite-difference scale per component: the largest root value,
        # else (every root 0 there) the typical seed
        scale = np.max(np.abs(np.array(deduped)), axis=0)
        seed_med = np.median(np.abs(seeds), axis=0)
        scale = np.where(scale > 0, scale, seed_med)
        _check_saddle_determined(R, x_sel, pure, expanded, scale)

    return {
        'mf_values':           selected,
        # Error estimate of each selected value (the last Newton correction;
        # a value snapped to an exact 0 carries its pre-snap size).
        'mf_errors':           selected_errs,
        # Full list (sorted, with per-root stability annotation when
        # classified) so callers can introspect every root.
        'mf_all_roots':        all_root_records,
        # Selectable subset — equal to all roots when stability is OFF,
        # else the stable-only subset.  Same sort order in both cases.
        'mf_stable_roots':     [r['values'] for r in selectable_records],
        # Unstable subset (only meaningful when stability was run).
        'mf_unstable_roots':   ([r['values'] for r in all_root_records
                                 if r['stable'] is False]
                                if stab_on else []),
        # Selected index into the selectable subset.
        'mf_index_used':       fixed_point_index,
        'state_var_order':     [v[0] for v in state_vars],
        'n_seeds_converged':   n_converged,
        # Diagnostic: was stability classification actually run?
        'stability_analysis':  stab_on,
        # Roots with a negative value in a field of undeclared domain:
        # outside the default search, not selectable (see the warning).
        'mf_unsearched_roots': [_build_root_dict(x) for x in unsearched],
    }


def _pure_integrator_rows(model, expanded) -> list:
    """The residual rows of the equations whose LHS is a pure time
    derivative (item 35): the LHS at Dt = 0 is identically zero as a
    symbolic expression, the parameters kept as symbols (the builder's
    test, ``ModelBuilder._lhs_vanishes_at_dt0``; an LHS such as
    ``(Dt + mu)*x`` is not one even where mu = 0).  Rows follow
    ``expanded`` (each equation once per population index)."""
    from api.model import ModelBuilder
    rows, k = [], 0
    for eq in model.get('equations', []):
        pop = eq['population']
        n = _pop_size(model, pop) if pop is not None else 1
        if ModelBuilder._lhs_vanishes_at_dt0(eq['lhs_text']):
            rows.extend(range(k, k + n))
        k += n
    return rows if k == len(expanded) else []


def _check_saddle_determined(R, x, rows, expanded, scale=None):
    """Raise when the equations do not determine the saddle at ``x``: the
    Jacobian of the residual (central differences) is singular.  Run only
    for models with a pure-integrator equation (``rows``), whose saddle
    condition 0 = RHS constrains the OTHER fields: when nothing pins the
    integrated field, every value of it is a root."""
    n = len(x)
    J = np.empty((n, n))
    for j in range(n):
        # relative step (the old floor 1.0 stepped far outside a field
        # measured in small units); ``scale``: the field's own size
        h = 1e-6 * max(abs(x[j]), float(scale[j]) if scale is not None
                       else 1.0)
        if h == 0.0:
            h = 1e-6
        e = np.zeros(n)
        e[j] = h
        rp, rm = R(x + e), R(x - e)
        # inconclusive: a non-finite value or a penalised point (R returns
        # _PENALTY where an equation fails) in the stencil
        if not (np.all(np.isfinite(rp)) and np.all(np.isfinite(rm))
                and np.max(np.abs(rp)) < _PENALTY
                and np.max(np.abs(rm)) < _PENALTY):
            return
        J[:, j] = (rp - rm) / (2.0 * h)
    # rank on the equilibrated Jacobian (rows, then columns, scaled to unit
    # norm), so that the units of a field or an equation do not decide it;
    # a zero row or column (the integrated field enters no equation) is
    # singular outright
    rn = np.linalg.norm(J, axis=1)
    sing = bool(np.any(rn == 0))
    if not sing:
        Js = J / rn[:, None]
        cn = np.linalg.norm(Js, axis=0)
        sing = bool(np.any(cn == 0))
        if not sing:
            sv = np.linalg.svd(Js / cn[None, :], compute_uv=False)
            sing = not sv[-1] > 1e-10 * sv[0]
    if not sing:
        return
    sv = np.linalg.svd(J, compute_uv=False)
    eqs = '; '.join(sorted({f'{expanded[k][0]} = {expanded[k][1]}'
                            for k in rows}))
    raise ValueError(
        f'solve_mean_field_dae: the mean-field equations do not determine '
        f'the saddle: their Jacobian at the root is singular (smallest '
        f'singular value {sv[-1]:.1e}, largest {sv[0]:.1e}).  The model has '
        f'pure-integrator equation(s) [{eqs}] (the LHS is a time derivative '
        f'only), whose stationarity condition 0 = RHS fixes the OTHER '
        f'fields, so the integrated field itself must be pinned by the rest '
        f'of the system (fed back into another equation).  Here nothing '
        f'pins it: every value is a fixed point.  Add a restoring term '
        f"(e.g. lhs='(Dt + eps)*h[i]') or the missing feedback.")


# ── Linear stability via generalized eigenvalue problem ──────────────


def _dt_offending_fields(resid, Dt_sym, sym_per_var) -> list:
    """State-variable names carrying the ``Dt``-degree ≥ 2 part of
    ``resid``, rendered ``'v[0]'`` / ``'x'`` for the error message."""
    from sage.all import SR, diff
    try:
        d2 = diff(diff(SR(resid), Dt_sym), Dt_sym).expand()
    except (AttributeError, TypeError, ValueError):
        return []
    hits = []
    for var, syms in sym_per_var.items():
        for idx, s in enumerate(syms):
            try:
                if d2.has(s):
                    hits.append(f'{var}[{idx}]' if len(syms) > 1 else var)
            except (AttributeError, TypeError, ValueError):
                continue
    return hits


def _check_equation_first_order_in_dt(resid, Dt_sym, model, eq, eq_no,
                                      i, pop, sym_per_var) -> None:
    """Raise ``NonlinearDtError`` if one declared equation is nonlinear
    in ``Dt``, naming the equation, its population slot, and the field(s)
    the offending term acts on."""
    from engine.core.dt_order import (check_first_order_in_dt,
                                      dt_degree_exceeds_one)

    # Fast path: this runs once per scalar residual, so the per-field
    # diagnostics below (O(M) symbolic ``has`` probes) must stay off it.
    if not dt_degree_exceeds_one(resid, Dt_sym):
        return

    where = (f"ModelBuilder.equation(lhs={eq['lhs_text']!r}, "
             f"rhs={eq['rhs_text']!r}"
             + (f", population={pop!r}" if pop is not None else '') + ')')
    fields = _dt_offending_fields(resid, Dt_sym, sym_per_var)
    subject = f'equation #{eq_no}'
    if pop is not None:
        subject += f' at population index i={i}'
    if fields:
        subject += f', field(s) {", ".join(sorted(set(fields)))}'
    display_subs = {
        str(s): (f'{var}[{idx}]' if len(syms) > 1 else var)
        for var, syms in sym_per_var.items()
        for idx, s in enumerate(syms)
    }
    check_first_order_in_dt(
        resid, Dt_sym,
        context=f"model {model.get('name', '<unnamed>')!r}, {where}",
        subject=subject,
        display_subs=display_subs,
    )


def linear_stability(
        model: dict,
        fundamental: dict,
        root: dict,
        *,
        verbose: bool = False,
) -> dict:
    """Classify the linear stability of a DAE fixed point.

    Linearizes the DAE ``LHS_k - RHS_k = 0`` around ``root`` and solves
    the generalized eigenvalue problem ``(σ·A + B)·δx = 0`` where

        A[k, j] = ∂²F_k / ∂Dt ∂x_j  |_{x*}
        B[k, j] = ∂F_k / ∂x_j       |_{x*, Dt=0}

    Algebraic equations (no ``Dt`` in LHS) have zero rows in ``A`` —
    those contribute "infinite eigenvalues" from the generalized
    eigenproblem and represent the implicit algebraic constraints; they
    get filtered out.  The finite eigenvalues are the linearized DAE's
    dynamical modes.

    Stability convention: the fixed point is stable iff every finite
    eigenvalue ``σ`` has ``Re(σ) < 0`` (perturbations decay).

    Parameters
    ----------
    model : dict
        Model dict with ``model['equations']`` populated.
    fundamental : dict
        Concrete parameter values.
    root : dict
        One element of ``solve_mean_field_dae(...)['mf_all_roots']``,
        i.e. ``{'<var>star': [vals]}``.
    verbose : bool
        Print A, B, and the eigenvalues.

    Returns
    -------
    dict with keys:
        ``'stable'`` (bool)
        ``'eigenvalues_finite'`` (np.ndarray, complex)
        ``'eigenvalues_all'`` (np.ndarray, complex; includes inf)
        ``'A'``, ``'B'`` (np.ndarray, real, M×M)
        ``'unstable_eigenvalues'`` (list of complex σ with Re ≥ 0)
    """
    from sage.all import SR, diff
    from sage.all import (tanh as _sage_tanh, sin as _sage_sin,
                          cos as _sage_cos, tan as _sage_tan,
                          exp as _sage_exp, log as _sage_log,
                          sqrt as _sage_sqrt, sinh as _sage_sinh,
                          cosh as _sage_cosh, pi as _sage_pi,
                          gamma as _sage_gamma,
                          factorial as _sage_factorial)
    import scipy.linalg

    if not model.get('equations'):
        raise ValueError(
            "linear_stability: model has no equations declared via "
            "ModelBuilder.equation(...)."
        )

    state_vars = _state_variables(model)
    slices = _state_slices(state_vars)
    keys = _saddle_keys(model, state_vars)

    # ── Build Sage symbols for every scalar state variable ───────
    flat_syms = []          # ordered list of Sage symbols
    sym_per_var = {}        # var_name → list of Sage symbols by pop idx
    flat_vals_at_root = []  # parallel list of root values
    for var, _, size in state_vars:
        sym_per_var[var] = []
        for i in range(size):
            s = SR.var(f'_mfdae_{var}_{i}')
            sym_per_var[var].append(s)
            flat_syms.append(s)
        flat_vals_at_root.extend(root[keys[var]])
    M = len(flat_syms)
    Dt_sym = SR.var('_mfdae_Dt')

    # Sage math namespace — these accept and return Sage SR
    # expressions so that user functions like ``tanh(g_gain[i]*v)``
    # are differentiable symbolically.
    sage_math_ns = {
        'tanh': _sage_tanh, 'sin':  _sage_sin,  'cos':  _sage_cos,
        'tan':  _sage_tan,  'exp':  _sage_exp,  'log':  _sage_log,
        'sqrt': _sage_sqrt, 'sinh': _sage_sinh, 'cosh': _sage_cosh,
        'abs':  abs,        'pi':   _sage_pi,
    }
    sage_special_ns = {'gamma': _sage_gamma, 'factorial': _sage_factorial}

    params_np = _numpy_params(model, fundamental)

    # ── Build phi callables that return Sage expressions ─────────
    phi_sym = {}
    for fn_spec in model.get('functions', []):
        args_text = fn_spec.get('args_text') or []
        expr_text = fn_spec.get('expression_text')
        if not args_text or expr_text is None or len(args_text) != 1:
            continue
        arg_name = args_text[0]
        pop = fn_spec.get('population')
        pop_size_val = _pop_size(model, pop)
        expr_py = _sage_to_python(expr_text)
        callables = []
        for i in range(pop_size_val):
            def phi_i(x_sym, _expr=expr_py, _arg=arg_name,
                      _i=i, _params=params_np):
                ns = {**sage_special_ns, **_params, **sage_math_ns,
                      _arg: x_sym, 'i': _i,
                      'sum': sum, '__builtins__': {}}
                return eval(_expr, ns)
            callables.append(phi_i)
        # Wrap in _PhiCallableList so scalar-mode ``f(v)`` works
        # alongside indexed ``f[i](v)``.  Mirrors the residual-eval
        # side of the same fix.
        phi_sym[fn_spec['name']] = _PhiCallableList(callables)

    # ── Build symbolic residuals ─────────────────────────────────
    idx_sets = _index_sets(model)
    # Wrap single-position state-var symbol lists so bare-name
    # arithmetic works in the symbolic eval too (mirroring the
    # ``_FieldScalar`` wrapping in ``model_compiler``).  Multi-
    # position lists stay raw — the user must use ``[i]`` access.
    from api.model_compiler import _FieldScalar
    sym_per_var_eval = {
        var: (_FieldScalar(arr) if len(arr) == 1 else arr)
        for var, arr in sym_per_var.items()
    }
    residuals_sym = []
    for eq_no, eq in enumerate(model['equations'], start=1):
        pop = eq['population']
        pop_size_val = _pop_size(model, pop) if pop is not None else 1
        lhs_py = _sage_to_python(eq['lhs_text'])
        rhs_py = _sage_to_python(eq['rhs_text'])
        for i in range(pop_size_val):
            ns = {
                **sage_special_ns,
                **params_np,
                **sym_per_var_eval,
                **phi_sym,
                **idx_sets,
                **sage_math_ns,
                'Dt':           Dt_sym,
                # Homogeneous-mode (k=0) linear stability of the
                # spatially-uniform saddle ⇒ Laplacian → 0.  (Stability
                # to k≠0 spatial perturbations is the pattern-formation
                # question, handled by the propagator's k-dependence in
                # Phase 2, not by MF saddle selection.)
                'Laplacian':    0,
                'i':            i,
                'sum':          sum,
                '__builtins__': {},
            }
            lhs = eval(lhs_py, ns)
            rhs = eval(rhs_py, ns)
            resid = SR(lhs - rhs)
            # First-order-in-Dt guard.  A term of Dt-degree >= 2 falls
            # out of BOTH halves of the pencil built below — B is
            # evaluated at Dt=0, and A differentiates once wrt Dt and
            # THEN evaluates at Dt=0 — so it would vanish from the
            # linearization without a trace.  Reject it loudly instead.
            _check_equation_first_order_in_dt(
                resid, Dt_sym, model, eq, eq_no, i, pop, sym_per_var)
            residuals_sym.append(resid)

    if len(residuals_sym) != M:
        raise ValueError(
            f"linear_stability: {len(residuals_sym)} equations after "
            f"population expansion vs {M} state-variable unknowns — "
            f"system is "
            f"{'under' if len(residuals_sym) < M else 'over'}-determined."
        )

    # ── Substitution dict for state vars at the root + Dt → 0 ────
    root_subs = {sym: float(val)
                 for sym, val in zip(flat_syms, flat_vals_at_root)}

    A_mat = np.zeros((M, M), dtype=float)
    B_mat = np.zeros((M, M), dtype=float)

    for k, F_k in enumerate(residuals_sym):
        # B[k, j] = ∂F_k/∂x_j at Dt=0, x=x*.
        F_k_alg = F_k.subs({Dt_sym: 0})
        for j, x_j in enumerate(flat_syms):
            d = diff(F_k_alg, x_j)
            B_mat[k, j] = float(d.subs(root_subs))

        # A[k, j] = ∂²F_k/∂Dt ∂x_j at x*.  Doing ∂Dt first reduces
        # the polynomial degree and avoids carrying a useless Dt
        # variable through the partial wrt x_j.
        F_k_dDt = diff(F_k, Dt_sym)
        for j, x_j in enumerate(flat_syms):
            d = diff(F_k_dDt, x_j)
            A_mat[k, j] = float(d.subs(root_subs).subs({Dt_sym: 0}))

    # ── Solve the generalized eigenvalue problem (σA + B) v = 0 ──
    # i.e. ``(-B) v = σ A v``.  scipy.linalg.eig(C, D) returns λ s.t.
    # C v = λ D v, so we call eig(-B, A) and the returned eigenvalues
    # ARE the σ values.
    sigmas, eigvecs = scipy.linalg.eig(-B_mat, A_mat)

    # Filter infinite / NaN eigenvalues (algebraic-constraint modes).
    finite_mask = np.isfinite(sigmas)
    sigmas_finite = sigmas[finite_mask]

    # Stability: every finite eigenvalue has strictly negative real part.
    # Use a small tolerance for the real-part comparison so eigenvalues
    # that are zero to roundoff don't get mis-classified -- relative to the
    # model's fastest rate read off the pencil (an absolute 1e-9 called a
    # stable model in slow time units marginal; the largest eigenvalue
    # itself is no scale: a stiff model's or an algebraic mode's QZ value
    # ~1e15 would swamp the slow modes).
    EIG_TOL = _EIG_RTOL * _rate_scale(A_mat, B_mat, sigmas_finite)
    stable = bool(
        sigmas_finite.size > 0
        and np.all(np.real(sigmas_finite) < -EIG_TOL)
    )
    unstable = [complex(s) for s in sigmas_finite
                if np.real(s) >= -EIG_TOL]

    if verbose:
        print(f'  A =\n{A_mat}')
        print(f'  B =\n{B_mat}')
        print(f'  finite eigenvalues σ: {sigmas_finite}')
        print(f'  stable: {stable}')

    return {
        'stable':                stable,
        'eigenvalues_finite':    sigmas_finite,
        'eigenvalues_all':       sigmas,
        'A':                     A_mat,
        'B':                     B_mat,
        'unstable_eigenvalues':  unstable,
    }


class UnstableSaddleError(ValueError):
    """The selected mean-field saddle is linearly unstable: there is no
    stationary state around it to expand about."""


def refuse_unstable_saddle(model, fundamental, mf):
    """Raise :class:`UnstableSaddleError` when the selected root of a model
    with ``.equation`` rows (stability analysis off) has a linearization
    eigenvalue with ``Re sigma > 0`` beyond round-off (relative to the
    model's fastest rate, as ``linear_stability`` judges it).  Without it an
    unstable selection surfaced as a ``PoleFreePropagatorError`` (the growing
    mode is not a retarded pole) when the analyzer is off.  Skipped where the
    linearization does not decide stability: no ``Dt`` rows, kernels (their
    images are collapsed to the integral), spatial models (q > 0 modes)."""
    if (not model.get('equations') or model.get('stability_analysis')
            or model.get('kernels') or model.get('spatial')):
        return
    roots = [r['values'] for r in (mf.get('mf_all_roots') or [])]
    sel = mf.get('mf_values')
    try:
        st = linear_stability(model, fundamental, sel)
    except Exception:                                    # noqa: BLE001
        return
    eigs = np.asarray(st['eigenvalues_finite'], dtype=complex)
    if not eigs.size:
        return
    tol = _EIG_RTOL * _rate_scale(st['A'], st['B'], eigs)
    grow = [complex(z) for z in eigs if z.real > tol]
    if not grow:
        return
    stable = []
    for i, r in enumerate(roots):
        try:
            if linear_stability(model, fundamental, r)['stable']:
                stable.append(i)
        except Exception:                                # noqa: BLE001
            pass
    shown = ', '.join(f'{z.real:.4g}{z.imag:+.4g}i' for z in grow)
    hint = (f'fixed_point_index={stable[0]}' + (f' (stable roots: '
            f'{stable})' if len(stable) > 1 else '')) if stable else (
            'no root found is stable: change the parameters')
    raise UnstableSaddleError(
        f'the selected mean-field saddle {sel} (fixed_point_index='
        f'{mf.get("mf_index_used")} of {len(roots)} root(s)) is linearly '
        f'unstable: eigenvalue(s) {shown} have Re > 0, so perturbations grow '
        f'and there is no stationary state around it.  Select a stable root '
        f'({hint}), or turn on stability_analysis(True) so that only stable '
        f'roots are selectable.')


# ── Compatibility wrapper for compute_cumulants ───────────────────────


def _check_selected_root(ft, model, num_params, index_used, mf_values,
                         errs=None):
    """Raise ``AssertionError`` when the selected saddle leaves the raw
    mean-field action sector (``ft._mf_sector_raw``: the (0,0), (1,0), (0,1)
    coefficients before the saddle substitution) visibly non-zero, by the
    relative rule shared with the expand-time check
    (:func:`engine.core.field_theory.mf_sector_residual`): a residual above
    ``MF_SECTOR_RTOL * sum of |terms|`` plus the residual the saddle's own
    error estimates ``errs`` (``{symbol: error}``) explain.  (The former
    absolute floor ``1e-7`` passed a wrong equation in small units.)  A
    coefficient with a symbol this run gives no value is skipped.  No-op
    without the raw sector (the proxy used inside ``expand``)."""
    raw = getattr(ft, '_mf_sector_raw', None)
    if not raw:
        return
    from engine.core.field_theory import (MF_SECTOR_ULPS, kernel_dc_gains_of,
                                          mf_sector_residual)
    from sage.all import SR
    gains = kernel_dc_gains_of(ft._ns, model)
    subs = dict(num_params)
    failures = []
    checked = []
    for key, poly in raw.items():
        try:
            items = poly.dict().items()
        except AttributeError:
            continue
        for exp_vec, coeff in items:
            c = SR(coeff)
            try:
                k_subs = gains.subs_dict(c)
            except ValueError:
                k_subs = None
            if k_subs:
                c = c.subs(k_subs)
            res = mf_sector_residual(c, subs, errs)
            if res is not None:
                checked.append((key, exp_vec, coeff) + tuple(res))
    # round-off allowance on the scale of the whole sector (a float residue
    # collapsed to a bare number has no terms of its own)
    floor = (MF_SECTOR_ULPS * np.finfo(float).eps
             * max([0.0] + [sc for *_, sc in checked if np.isfinite(sc)]))
    for key, exp_vec, coeff, resid, tol, scale in checked:
        if not np.isfinite(resid) or resid > tol + floor:
            failures.append((key, exp_vec, coeff, resid, scale))
    if failures:
        lines = [
            "This model's mean-field saddle does not satisfy its own "
            f"equations at the selected root (fixed_point_index="
            f"{index_used}: {mf_values}) — the MF action sector is non-zero "
            "there.  The equations (.equation / .set_mf_equation) do not "
            "match the action's deterministic drift at this root and these "
            "parameters.",
            "Diagnostics (internal):"]
        for key, exp_vec, coeff, resid, scale in failures:
            lines.append(f'  bigrade={key}  monomial_exponents={exp_vec}\n'
                         f'    coefficient: {coeff}\n'
                         f'    residual at the root: {resid:.3e} '
                         f'(terms of size {scale:.3e})')
        raise AssertionError('\n'.join(lines))


def solve_mean_field_dae_compat(
        ft,
        model: dict,
        fundamental: dict,
        *,
        fixed_point_index: int = 0,
        n_starts: int = 64,
        seed_box: Optional[dict[str, tuple[float, float]]] = None,
        verbose: bool = True,
        rtol: float = 1e-6,
        atol: float = 0.0,
) -> dict:
    """The mean-field solve of every model (``api._mean_field.
    solve_mean_field`` is its front door): :func:`solve_mean_field_dae` on
    the model's ``.equation`` rows or its ``.set_mf_equation`` saddle
    equations, the selected root checked against the action's mean-field
    sector at this run's parameters.

    Returns ``num_params`` (every parameter and saddle SR symbol on
    ``ft._ns`` mapped to its value, plus any other entry of the model's
    ``mf_bg_conditions`` evaluated at the saddle), ``num_params_err``,
    ``param_subs``, ``saddle_values`` / ``mf_values`` (``{saddle name:
    [values]}``) and the root census (``mf_all_roots``, ``mf_index_used``,
    ``state_var_order``, ``n_seeds_converged``, ...).
    """
    from sage.all import SR

    # 1. Run the multi-start Newton; pick the requested root.
    dae_result = solve_mean_field_dae(
        model, fundamental,
        n_starts=n_starts,
        fixed_point_index=fixed_point_index,
        seed_box=seed_box,
        rtol=rtol, atol=atol,
        verbose=False,
        ns=ft._ns,
    )

    mf_values = dae_result['mf_values']

    if verbose:
        n_roots = len(dae_result['mf_all_roots'])
        used = dae_result['mf_index_used']
        print(f'  DAE solver found {n_roots} fixed point(s); '
              f'using fixed_point_index={used}')
        for i, r in enumerate(dae_result['mf_all_roots']):
            tag = ' ← selected' if i == used else ''
            print(f'    root[{i}]: {r}{tag}')

    # 2. Parameter values on the namespace's SR symbols.
    from api._mean_field import complete_parameters
    fundamental = complete_parameters(model, fundamental,
                                     'mean-field solve', strict=False)
    param_subs = _param_subs(model, fundamental, _pop_sizes(model, ft._ns))

    # 3. Bake saddle values into num_params using ft._ns.<saddle_name>.
    num_params = dict(param_subs)
    num_params_err = {}
    mf_errs = dae_result.get('mf_errors') or {}
    for saddle_name, vals in mf_values.items():
        sr_array = getattr(ft._ns, saddle_name, None)
        if sr_array is None:
            continue
        errs = mf_errs.get(saddle_name) or [0.0] * len(vals)
        for i, v in enumerate(vals):
            num_params[sr_array[i]] = float(v)
            num_params_err[sr_array[i]] = float(errs[i])

    # 3b. The selected root must zero the action's mean-field sector at the
    # parameters of THIS run.  ``FieldTheory.expand`` checks the sector at
    # the model's default point and root 0 only (and not at all on an
    # expand-cache hit), so an equation that shares that root but no other
    # was accepted at fixed_point_index >= 1.
    _check_selected_root(ft, model, num_params,
                         dae_result['mf_index_used'], mf_values,
                         num_params_err)

    # 4. Any other entry of the saddle dict (a formal function's value at
    # the saddle, ``phi0_i -> nstar_i``) evaluated at the solved saddle.
    if not model.get('equations'):
        mf_bg = model.get('mf_bg_conditions', lambda ns: {})(ft._ns)
        from api._mean_field import _kernel_saddle_subs
        rest = [k for k in mf_bg if k not in num_params]
        ksubs = _kernel_saddle_subs(ft._ns, model,
                                    _kernel_symbols(model, ft._ns),
                                    [mf_bg[k] for k in rest])
        for lhs in rest:
            try:
                num_params[lhs] = float(
                    SR(mf_bg[lhs]).subs(ksubs).subs(num_params))
            except (TypeError, ValueError):
                pass              # still symbolic: left out

    return {
        'num_params':         num_params,
        # error estimate of each saddle value in num_params (the sector
        # checks allow the residual these explain)
        'num_params_err':     num_params_err,
        'param_subs':         param_subs,
        'saddle_values':      dict(mf_values),
        # DAE-specific extras (consumed by compute_cumulants to surface
        # them on the returned dict via ``th['mf_all_roots']`` etc.).
        # ``mf_all_roots`` carries STABILITY ANNOTATION per root —
        # ``[{'values': {...}, 'stable': bool, 'eigenvalues_finite':
        # np.array}, ...]``.  ``mf_stable_roots`` is the same subset
        # without unstable entries and stripped to bare value dicts.
        'mf_values':          dict(mf_values),
        'mf_all_roots':       list(dae_result['mf_all_roots']),
        'mf_stable_roots':    list(dae_result['mf_stable_roots']),
        'mf_unstable_roots':  list(dae_result['mf_unstable_roots']),
        'mf_index_used':      dae_result['mf_index_used'],
        'state_var_order':    list(dae_result['state_var_order']),
        'n_seeds_converged':  dae_result['n_seeds_converged'],
    }

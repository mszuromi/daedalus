"""
DAE-based mean-field solver — multi-root + linear-stability ready.

Consumes ``model['equations']`` (declared via
``TheoryBuilder.equation(lhs=..., rhs=..., population=...)``) and finds
ALL fixed points of the algebraic system obtained by setting ``Dt → 0``
on every equation's LHS.  Uses multi-start Newton (scipy ``optimize.root``
with the ``hybr`` Powell hybrid method), deduplicates clusters, filters
out non-real solutions, sorts by the first declared physical field's
first population index, and selects ``fixed_point_index``-th root.

Companion ``linear_stability(model, fundamental, root)`` (step c, lands
separately) builds the generalized-eigenvalue problem at a given root
using the same equation parsing infrastructure here.

This module is INTENTIONALLY decoupled from
``pipeline/_mean_field.py`` — the legacy iteration solver still runs
for theories without ``.equation(...)`` declarations.  Routing happens
in ``compute_cumulants`` (step d).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import root as _scipy_root


# Math functions that may appear in user equation text (LHS / RHS of
# .equation(...), or inside .define_function(...) expressions).
# Drawn from numpy so they work on both scalars and numpy arrays.
_MATH_NS = {
    'tanh': np.tanh, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'exp':  np.exp,  'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
    'sinh': np.sinh, 'cosh': np.cosh,
    'pi':   float(np.pi),
}


def _sage_to_python(text: str) -> str:
    """Translate Sage-syntax expression text to plain-Python syntax
    so it parses cleanly under ``eval``.

    The only difference that matters in practice is the power
    operator: Sage and TheoryBuilder docstrings use ``^`` (which Sage
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
        # internal name for theories that didn't declare one.
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


def _make_phi_callables(model: dict, params: dict) -> dict[str, list]:
    """For each declared function (``define_function('phi', ...)``),
    return a list of callables — one per population position — that
    take a single state-variable value and return the function's value.

    The callable is built by Python ``eval`` on the function's
    expression text with the argument-name bound to the input value
    and ``i`` bound to the population index.

    Returns a dict ``{function_name: [callable_for_pos_0, ...]}``.
    Currently only single-arg functions are supported (the framework's
    main use case for ``phi``).  Multi-arg functions are skipped — the
    DAE solver only needs the saddle value, and indexed multi-arg
    functions used in CGF terms don't appear in equation RHSs.
    """
    out = {}
    for fn_spec in model.get('functions', []):
        # Read text-form spec preserved by TheoryBuilder (see
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
                ns = {**_params, **_MATH_NS,
                      _arg: x_val, 'i': _i,
                      'sum': sum, '__builtins__': {}}
                return eval(_expr, ns)
            callables.append(phi_i)
        out[fn_spec['name']] = callables
    return out


# ── Parameter dict → numpy-ized for eval ─────────────────────────────


def _numpy_params(model: dict, fundamental: dict) -> dict:
    """Convert ``fundamental`` to a dict of numpy arrays so that
    expressions like ``w[i,j]`` (2-D indexing) and ``Em[i]`` (1-D) work
    under plain Python ``eval``.
    """
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
    for eq in equations:
        pop = eq['population']
        pop_size_val = _pop_size(model, pop) if pop is not None else 1
        lhs_py = _sage_to_python(eq['lhs_text'])
        rhs_py = _sage_to_python(eq['rhs_text'])
        for i in range(pop_size_val):
            expanded.append((lhs_py, rhs_py, i, pop))

    def R(x: np.ndarray) -> np.ndarray:
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
            **params_np,
            **state,
            **phi_callables,
            **index_sets,
            **_MATH_NS,
            'Dt':           0,
            'sum':          sum,
            '__builtins__': {},
        }
        out = np.empty(len(expanded), dtype=float)
        for k, (lhs_text, rhs_text, i, pop) in enumerate(expanded):
            ns['i'] = i
            try:
                lhs = eval(lhs_text, ns)
                rhs = eval(rhs_text, ns)
                out[k] = float(lhs - rhs)
            except (ValueError, ZeroDivisionError, OverflowError,
                    FloatingPointError):
                # Penalize: send the solver away from this region.
                out[k] = 1e10
        return out

    return R, expanded


# ── Initial-guess sampling ───────────────────────────────────────────


def _seed_box_default(state_vars, model, fundamental):
    """Per-variable default sampling box derived from the variable's
    declared ``domain`` and a heuristic scale.

    Heuristic: scale = max(|param value|) over all declared parameters
    (fallback 1.0).  For ``domain='positive'`` we sample uniformly in
    ``[0, 5·scale]``; for any other declared domain (``'real'`` or
    unspecified) we sample in ``[-3·scale, 3·scale]``.

    Returns ``{var_name: (low, high)}``.
    """
    scale = 0.0
    for pname, pval in fundamental.items():
        arr = np.asarray(pval, dtype=float).ravel()
        if arr.size:
            scale = max(scale, float(np.max(np.abs(arr))))
    if scale <= 0.0:
        scale = 1.0

    boxes = {}
    field_specs = {f['name']: f for f in model.get('physical_fields', [])}
    for var, _, _ in state_vars:
        domain = (field_specs.get(var, {}).get('domain')
                  or 'positive')   # default: positive (Hawkes-like)
        if domain == 'positive':
            boxes[var] = (0.0, 5.0 * scale)
        else:
            boxes[var] = (-3.0 * scale, 3.0 * scale)
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


# ── Root deduplication ──────────────────────────────────────────────


def _dedup_roots(roots: list[np.ndarray], rtol=1e-6, atol=1e-10
                 ) -> list[np.ndarray]:
    """Cluster roots within ``rtol/atol``, keep one representative
    each.  Order of returned list follows discovery order (sorting
    happens in the outer caller)."""
    out = []
    for r in roots:
        merged = False
        for kept in out:
            diff = np.max(np.abs(r - kept))
            scale = max(np.max(np.abs(r)), np.max(np.abs(kept)), 1.0)
            if diff < atol + rtol * scale:
                merged = True
                break
        if not merged:
            out.append(r)
    return out


# ── Top-level solver ─────────────────────────────────────────────────


def solve_mean_field_dae(
        model: dict,
        fundamental: dict,
        *,
        n_starts: int = 64,
        fixed_point_index: int = 0,
        seed_box: Optional[dict[str, tuple[float, float]]] = None,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        verbose: bool = False,
        rng_seed: int = 0,
) -> dict:
    """Solve the MF DAE system declared via
    ``TheoryBuilder.equation(...)`` calls.

    Parameters
    ----------
    model : dict
        Theory dict from ``TheoryBuilder.build()``.  Must contain
        ``model['equations']`` (non-empty) — equation residuals
        ``LHS - RHS = 0``, ``Dt`` allowed on LHS only.
    fundamental : dict
        Concrete parameter values keyed by parameter name.
    n_starts : int, default 64
        Number of random initial guesses for multi-start Newton.
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
        Tolerances for clustering converged roots.
    verbose : bool
        Print a one-line summary per seed (for debugging).
    rng_seed : int
        Seed for the initial-guess RNG (kept reproducible).

    Returns
    -------
    dict with keys:
        ``'mf_values'`` — selected root as ``{<var>star: [vals]}``.
        ``'mf_all_roots'`` — full sorted list of distinct roots, each
            a dict shaped like ``mf_values``.
        ``'mf_index_used'`` — int, actual index used (after clamping).
        ``'state_var_order'`` — list of state variable names in
            declaration order (i.e. the sort-key origin).
        ``'n_seeds_converged'`` — diagnostic: how many seeds led to a
            valid root before dedup.
    """
    if not model.get('equations'):
        raise ValueError(
            "solve_mean_field_dae: model has no equations declared via "
            "TheoryBuilder.equation(...).  Use the legacy iteration "
            "solver in pipeline._mean_field instead."
        )

    state_vars = _state_variables(model)
    if not state_vars:
        raise ValueError(
            "solve_mean_field_dae: no physical fields declared — "
            "there's nothing to solve for."
        )
    slices = _state_slices(state_vars)
    total = sum(size for _, _, size in state_vars)

    params_np = _numpy_params(model, fundamental)
    phi_callables = _make_phi_callables(model, params_np)
    idx_sets = _index_sets(model)

    R, expanded = _build_residual(
        model, params_np, phi_callables, idx_sets, state_vars, slices)

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

    raw_roots: list[np.ndarray] = []
    n_converged = 0
    for s in range(n_starts):
        x0 = seeds[s]
        try:
            sol = _scipy_root(R, x0, method='hybr')
        except (ValueError, ZeroDivisionError, OverflowError,
                FloatingPointError):
            continue
        if not sol.success:
            continue
        # Re-evaluate residual to confirm; scipy sometimes reports
        # success but with a residual above its internal tolerance.
        resid = R(sol.x)
        if np.max(np.abs(resid)) > 1e-7:
            continue
        # Filter complex-or-NaN.
        if not np.all(np.isfinite(sol.x)):
            continue
        n_converged += 1
        raw_roots.append(np.asarray(sol.x, dtype=float))
        if verbose:
            print(f"  seed {s:3d}: converged to {sol.x}")

    deduped = _dedup_roots(raw_roots, rtol=rtol, atol=atol)

    # Sort ascending by first declared physical field's first index.
    first_var = state_vars[0][0]
    first_start = slices[first_var][0]
    deduped.sort(key=lambda r: float(r[first_start]))

    if not deduped:
        raise ValueError(
            f"solve_mean_field_dae: no MF fixed point found from "
            f"{n_starts} seeds.  Try increasing n_starts, passing "
            f"explicit seed_box, or checking the equations for typos."
        )

    def _build_root_dict(x):
        return {f'{var}star': x[slices[var][0]:slices[var][1]].tolist()
                for var, _, _ in state_vars}

    # Classify every root for stability inline so ``fixed_point_index``
    # can index over the STABLE subset only (the unstable saddles are
    # not physically expandable — surfacing them in
    # ``mf_all_roots`` for inspection but not as a selectable
    # expansion point).
    all_root_records = []
    for x in deduped:
        rd = _build_root_dict(x)
        try:
            stab = linear_stability(model, fundamental, rd, verbose=False)
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

    stable_records = [r for r in all_root_records if r['stable']]

    if not stable_records:
        raise ValueError(
            f"solve_mean_field_dae: found {len(all_root_records)} fixed "
            f"point(s) but NONE were classified linearly stable.  Every "
            f"root has at least one finite eigenvalue with non-negative "
            f"real part — the diagrammatic expansion is undefined at all "
            f"of them.  Inspect the eigenvalues via "
            f"``linear_stability(model, fundamental, root)`` and revisit "
            f"the equations or parameters."
        )

    # fixed_point_index now indexes over STABLE roots, sorted ascending
    # by the same key (first physical field's first index).  Clamp +
    # warn on out-of-range exactly as before, but using the stable
    # subset's length.
    n_stable = len(stable_records)
    if fixed_point_index < 0 or fixed_point_index >= n_stable:
        clamped = max(0, min(fixed_point_index, n_stable - 1))
        warnings.warn(
            f"solve_mean_field_dae: requested fixed_point_index="
            f"{fixed_point_index} but only {n_stable} stable root(s) "
            f"found (out of {len(all_root_records)} total); using "
            f"index {clamped}.",
            stacklevel=2,
        )
        fixed_point_index = clamped

    selected = stable_records[fixed_point_index]['values']

    return {
        'mf_values':           selected,
        # Full list (sorted, with stability annotation per entry) so
        # callers can introspect unstable roots if desired.
        'mf_all_roots':        all_root_records,
        # Stable subset only, in the same sort order.
        'mf_stable_roots':     [r['values'] for r in stable_records],
        'mf_unstable_roots':   [r['values'] for r in all_root_records
                                if not r['stable']],
        # ``mf_index_used`` is the index into ``mf_stable_roots`` — the
        # selectable enumeration.
        'mf_index_used':       fixed_point_index,
        'state_var_order':     [v[0] for v in state_vars],
        'n_seeds_converged':   n_converged,
    }


# ── Linear stability via generalized eigenvalue problem ──────────────


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
        Theory dict with ``model['equations']`` populated.
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
                          cosh as _sage_cosh, pi as _sage_pi)
    import scipy.linalg

    if not model.get('equations'):
        raise ValueError(
            "linear_stability: model has no equations declared via "
            "TheoryBuilder.equation(...)."
        )

    state_vars = _state_variables(model)
    slices = _state_slices(state_vars)

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
        flat_vals_at_root.extend(root[f'{var}star'])
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
                ns = {**_params, **sage_math_ns,
                      _arg: x_sym, 'i': _i,
                      'sum': sum, '__builtins__': {}}
                return eval(_expr, ns)
            callables.append(phi_i)
        phi_sym[fn_spec['name']] = callables

    # ── Build symbolic residuals ─────────────────────────────────
    idx_sets = _index_sets(model)
    residuals_sym = []
    for eq in model['equations']:
        pop = eq['population']
        pop_size_val = _pop_size(model, pop) if pop is not None else 1
        lhs_py = _sage_to_python(eq['lhs_text'])
        rhs_py = _sage_to_python(eq['rhs_text'])
        for i in range(pop_size_val):
            ns = {
                **params_np,
                **{var: sym_per_var[var] for var in sym_per_var},
                **phi_sym,
                **idx_sets,
                **sage_math_ns,
                'Dt':           Dt_sym,
                'i':            i,
                'sum':          sum,
                '__builtins__': {},
            }
            lhs = eval(lhs_py, ns)
            rhs = eval(rhs_py, ns)
            residuals_sym.append(SR(lhs - rhs))

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
    # that are zero to roundoff don't get mis-classified.
    EIG_TOL = 1e-9
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


# ── Compatibility wrapper for compute_cumulants ───────────────────────


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
        atol: float = 1e-10,
) -> dict:
    """Drop-in replacement for ``pipeline._mean_field.solve_mean_field``
    when ``model['equations']`` is populated.

    Returns the same keys the legacy solver does (``nstar_vals``,
    ``vstar_vals``, ``num_params``, ``param_subs``, ``phi_deriv_vals``,
    ``saddle_values``) PLUS the DAE-specific extras (``mf_all_roots``,
    ``mf_index_used``, ``state_var_order``, ``n_seeds_converged``).

    ``num_params`` is built by mapping the Sage SR vars on ``ft._ns``
    (parameter symbols + saddle arrays) to their concrete values, so
    downstream code (``compute_poles_and_residues`` etc.) consumes it
    identically to the legacy solver's output.

    ``phi_deriv_vals`` is left as ``{}`` for the moment — heterogeneous
    legacy paths also return it empty, and the propagator builds its
    derivatives symbolically.  If a downstream consumer ends up needing
    explicit derivatives at the saddle, we'll populate it here.
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

    # 2. Build param_subs in the legacy Sage-symbol convention.
    pop_size_map = getattr(ft._ns, '_pop_size', None) or {}
    # Fall back: walk model['populations'] directly when ns doesn't
    # carry _pop_size (single-pop legacy path).
    if not pop_size_map and model.get('populations'):
        pop_size_map = {p['name']: int(p.get('size', 1))
                        for p in model['populations']}

    param_subs: dict = {}
    for pspec in model.get('parameters', []):
        pname = pspec['name']
        if pname not in fundamental:
            continue
        val = fundamental[pname]
        ib = pspec.get('indexed_by')
        if ib:
            if len(ib) == 2:
                n_rows = pop_size_map.get(ib[0], 1)
                n_cols = pop_size_map.get(ib[1], 1)
                for i in range(n_rows):
                    for j in range(n_cols):
                        sym = SR.var(f'{pname}{i+1}{j+1}')
                        param_subs[sym] = float(val[i][j])
            elif len(ib) == 1:
                n = pop_size_map.get(ib[0], 1)
                for i in range(n):
                    sym = SR.var(f'{pname}{i+1}')
                    param_subs[sym] = float(val[i])
            else:
                param_subs[SR.var(pname)] = float(val)
        elif isinstance(val, (list, tuple)):
            # Un-indexed but vector value (legacy single-pop).
            for i, v in enumerate(val):
                param_subs[SR.var(f'{pname}{i+1}')] = float(v)
        else:
            param_subs[SR.var(pname)] = float(val)

    # 3. Bake saddle values into num_params using ft._ns.<saddle_name>.
    num_params = dict(param_subs)
    for saddle_name, vals in mf_values.items():
        sr_array = getattr(ft._ns, saddle_name, None)
        if sr_array is None:
            continue
        for i, v in enumerate(vals):
            num_params[sr_array[i]] = float(v)

    # 4. Legacy-compatible concatenated saddle vectors.
    nstar_concat = list(mf_values.get('nstar', []))
    vstar_concat = list(mf_values.get('vstar', []))

    return {
        'nstar_vals':         nstar_concat,
        'vstar_vals':         vstar_concat,
        'phi_deriv_vals':     {},
        'num_params':         num_params,
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

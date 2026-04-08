"""
msrjd.integration.time_domain.final_integral
=============================================
Vertex-time integration on a tree-level diagram via explicit numerical
quadrature.

MVP scope
---------
Only tree-level (loop_number == 0) typed diagrams are handled in the
initial build. Tree-level is the right proving ground for the Phase J
evaluation layer because it exercises:

- time-domain retarded propagator lookup per edge,
- polytope extraction from explicit Heaviside factors,
- vertex-time integration over the retarded polytope,
- global translation invariance / origin pinning,
- numerical dispatch,

**without** needing the kernel reduction / caching / contraction
machinery (Phases 3-5 of the hybrid pipeline). Loop cases are deferred
to Extension 1.

Integration strategy (numerical, not symbolic)
----------------------------------------------
The integrand for a tree-level diagram is

    combined_prefactor · ∏_{edges (u,v)} G^R_{phys, resp}(t_v - t_u)

where each retarded propagator factor carries an implicit Heaviside at
`t_v - t_u`. Rather than hand the whole thing to SageMath's symbolic
`integrate()` (which returns unevaluated integrals when bounds depend
on sign of symbolic external times), we:

1. **Strip the Heaviside factors** at construction time. `G_t_entry`
   is called with `include_heaviside=False` so the per-edge factor is a
   pure exponential in the vertex-time differences, and the Heaviside
   arguments `dt_e = t_v - t_u` are collected separately as linear
   inequality constraints `dt_e > 0` on the vertex-time variables.
2. **JIT-compile the stripped integrand** via SageMath's
   `fast_callable` over the `CDF` domain so evaluation is a C-level
   operation on concrete floats, taking `(s_1, ..., s_m, t_1, ..., t_k)`
   as positional arguments.
3. **Extract linear coefficients** from each constraint expression
   (coefficients on integration variables, on external time variables,
   and the constant) so the constraints can be resolved to concrete
   numeric inequalities in `s` once external times are supplied.
4. **Compute the polytope bounds** on each integration variable. For
   the linear Hawkes tree case (m = 1) this is intersection of
   half-lines — a single `[L, U]` interval. For m >= 2 the polytope is
   represented by nested bound functions so we can use
   `scipy.integrate.nquad`.
5. **Call `scipy.integrate.quad` / `nquad`** on the `fast_callable`
   integrand with the resolved polytope bounds. Real and imaginary
   parts are integrated separately to support complex-valued residues.

The public entry point `integrate_tree_diagram` returns a Python
callable `contribution(*ext_time_values) -> complex`, not an SR
expression. Downstream code (the orchestrator, the notebook cells, the
tests) calls this directly without going through any symbolic layer.
"""

import math

from sage.all import SR, fast_callable, CDF

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
)


# ───────────────────────────────────────────────────────────────────────
# Tree-level vertex-time integration
# ───────────────────────────────────────────────────────────────────────

def integrate_tree_diagram(
    typed_diagram,
    representative_ir,
    propagator_data,
    combined_prefactor,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    timeout_sec=30,  # unused on the numerical path; kept for API compat.
):
    r"""
    Vertex-time integration for a TREE-LEVEL typed diagram, evaluated
    via explicit numerical quadrature.

    Returns a dict whose `contribution` value is a Python callable

        f(*ext_time_values) -> complex

    taking `k` positional arguments (one per entry in `ext_time_vars`,
    in the same order). If `origin_leaf_idx` is not None, the value
    supplied at that position is ignored (it was pinned to zero during
    integrand construction).

    Parameters
    ----------
    typed_diagram : TypedDiagram
        Must have `loop_number == 0` (asserted below). Needed for the
        prediagram `D`, leaf list, and `propagator_indices`.
    representative_ir : dict
        Output of `build_integrand_stationary`. Used only for the
        loop_number sanity check.
    propagator_data : dict
        Standard pipeline propagator dict with keys `'pole_vals'` and
        `'C_mats'`.
    combined_prefactor : SR or numeric
        Sum of scalar prefactors over diagrams in the kernel group.
    ext_time_vars : list of SR
        `k` external time variables, one per leaf (in `leaves` order).
    num_params : dict or None
        Numerical parameter substitutions for the propagator matrix AND
        the combined prefactor. Required if either the propagator
        entries or the prefactor contain free symbolic parameters — the
        JIT-compiled integrand cannot be built until every symbol
        except the integration variables and external times has been
        substituted.
    origin_leaf_idx : int or None
        Which external leaf's time to pin to zero. Default 0. Pass None
        to leave all external times free — the returned callable will
        then depend on all `k` external times.
    timeout_sec : int
        Unused on the numerical quadrature path (kept for API
        compatibility with earlier symbolic-integration builds).

    Returns
    -------
    dict with keys:
        'status' : 'ok' | 'empty_polytope' | 'failed'
        'contribution' : callable
            `f(*ext_time_values) -> complex`. For status != 'ok' this
            may be `None`.
        'integration_vars' : list of SR
            The non-leaf vertex time symbols that were integrated out.
        'stripped_integrand' : SR
            The symbolic integrand WITHOUT the Heaviside factors, for
            debugging.
        'constraints' : list of SR
            One SR expression per edge; each expression `dt_e` is the
            linear combination `t_head - t_tail` that must be positive
            for the integrand to be nonzero.
    """
    loop_number = representative_ir.get('loop_number', 0)
    assert loop_number == 0, (
        "integrate_tree_diagram only handles tree-level diagrams; "
        f"got loop_number = {loop_number}. Loop cases must go through "
        "the contraction path (not implemented in the MVP)."
    )

    D = typed_diagram.prediagram[0]
    leaves = list(typed_diagram.prediagram[2])
    leaf_set = set(leaves)

    if len(ext_time_vars) != len(leaves):
        raise ValueError(
            f"ext_time_vars has length {len(ext_time_vars)} but "
            f"the diagram has {len(leaves)} leaves."
        )

    # ── 1. Numerical G(t) matrix ─────────────────────────────────
    t_sym = SR.var('_t_td_')
    G_t_matrix = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

    # ── 2. Assign a time symbol to every vertex ──────────────────
    vertex_time = {}
    for j, lf in enumerate(leaves):
        t_ext = ext_time_vars[j]
        if origin_leaf_idx is not None and j == origin_leaf_idx:
            t_ext = SR(0)
        vertex_time[lf] = t_ext

    internal_vertices = [v for v in D.vertices() if v not in leaf_set]
    integration_vars = []
    for v in internal_vertices:
        s_v = SR.var(f's_v{v}_td_', latex_name=rf's_{{v_{{{v}}}}}')
        vertex_time[v] = s_v
        integration_vars.append(s_v)

    # ── 3. Build stripped integrand + collect constraints ────────
    stripped = SR(1)
    constraints = []  # list of SR expressions; each must be > 0
    for (u, v, lbl) in D.edges():
        ri, pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        dt = vertex_time[v] - vertex_time[u]
        # Edge factor WITHOUT heaviside: pure exponential
        factor = G_t_entry(G_t_matrix, pi, ri, dt, include_heaviside=False)
        stripped = stripped * factor
        constraints.append(SR(dt))

    # Apply the combined prefactor (possibly with num_params)
    if combined_prefactor is not None:
        cp = SR(combined_prefactor)
        if num_params:
            cp = cp.subs(num_params)
        stripped = cp * stripped

    if num_params:
        # Safety net: substitute num_params into the integrand in case
        # any free parameter slipped through the propagator path.
        stripped = stripped.subs(num_params)

    # ── Flatten the integrand into an explicit sum of single
    # exponentials.
    #
    # `stripped` is a product of edge factors, each of which is itself a
    # sum of exponentials (one per pole of the kernel matrix). Left in
    # that product form, `fast_callable` evaluates each factor
    # separately: at large negative vertex-time values, individual
    # factors grow like exp(|α_i|·|s|) and their pairwise products can
    # overflow IEEE double precision BEFORE the causal suppression
    # factor exp(−α_total·|s|) has a chance to bring the product back
    # into range. scipy.integrate.quad samples at arbitrarily negative
    # `s` as part of its adaptive quadrature over (−∞, L], so it will
    # hit any such overflow and return NaN.
    #
    # The fix is to distribute the product into a sum of terms where
    # each term is `C · exp(α·s + …)` — a single exponential whose
    # coefficient `α > 0` (guaranteed by retarded causality). Sage's
    # `.expand()` handles this: `(A·e^a + B)·(C·e^c + D)·e^g` becomes
    # `A·C·e^(a+c+g) + A·D·e^(a+g) + B·C·e^(c+g) + B·D·e^g`, and each
    # summand is numerically stable on `(−∞, L]`.
    try:
        stripped = stripped.expand()
    except Exception:
        pass

    # ── 4. Resolve which external times remain free ──────────────
    # The pinned one was set to SR(0) in vertex_time so it cannot
    # appear in either the stripped integrand or the constraints.
    free_ext_idx = [
        j for j in range(len(ext_time_vars))
        if (origin_leaf_idx is None or j != origin_leaf_idx)
    ]

    # Variables for the fast_callable: [s_0, ..., s_{m-1}, t_{free_0}, ...]
    fc_vars = list(integration_vars) + [ext_time_vars[j] for j in free_ext_idx]

    # Check that nothing exotic remains in the stripped integrand.
    stripped_free_vars = set(stripped.variables()) if stripped != 0 else set()
    unexpected = stripped_free_vars - set(fc_vars)
    if unexpected:
        return {
            'status': 'failed',
            'contribution': None,
            'integration_vars': integration_vars,
            'stripped_integrand': stripped,
            'constraints': constraints,
            'reason': (
                f"stripped integrand contains unexpected free symbols "
                f"{unexpected}; pass them via num_params."
            ),
        }

    # ── 5. JIT-compile the integrand over CDF ────────────────────
    try:
        integrand_callable = fast_callable(stripped, vars=fc_vars, domain=CDF)
    except Exception as exc:
        return {
            'status': 'failed',
            'contribution': None,
            'integration_vars': integration_vars,
            'stripped_integrand': stripped,
            'constraints': constraints,
            'reason': f"fast_callable failed: {exc}",
        }

    # ── 6. Extract linear coefficients from each constraint ──────
    # Each constraint is a linear expression in (integration_vars, free ext
    # time vars). The caller supplies numeric values for the free ext
    # time vars; the constraint then reduces to a linear constraint in
    # the integration variables alone.
    constraint_data = []
    for c in constraints:
        c = SR(c)
        try:
            a_int = [float(c.coefficient(v)) for v in integration_vars]
            a_ext = [float(c.coefficient(ext_time_vars[j])) for j in free_ext_idx]
            zero_subs = {v: 0 for v in list(integration_vars)
                         + [ext_time_vars[j] for j in free_ext_idx]}
            c0 = float(c.subs(zero_subs))
        except (TypeError, ValueError) as exc:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': stripped,
                'constraints': constraints,
                'reason': (
                    f"constraint {c} is not linear in "
                    f"(integration_vars, ext_time_vars): {exc}"
                ),
            }
        constraint_data.append((a_int, a_ext, c0))

    m = len(integration_vars)

    # ── 7. Build the callable ────────────────────────────────────
    def contribution(*ext_time_values):
        if len(ext_time_values) != len(ext_time_vars):
            raise ValueError(
                f"contribution() expects {len(ext_time_vars)} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        # Select the free external time values (pinned slot is ignored)
        free_vals = [float(ext_time_values[j]) for j in free_ext_idx]

        # Resolve each constraint at the given ext_time_values
        # (a_int · s + c_eff > 0)
        resolved = []
        for (a_int, a_ext, c0) in constraint_data:
            c_eff = c0 + sum(a_ext[i] * free_vals[i]
                             for i in range(len(a_ext)))
            resolved.append((list(a_int), c_eff))

        return _integrate_polytope(
            integrand_callable, resolved, free_vals, m
        )

    return {
        'status': 'ok',
        'contribution': contribution,
        'integration_vars': integration_vars,
        'stripped_integrand': stripped,
        'constraints': constraints,
    }


# ───────────────────────────────────────────────────────────────────────
# Polytope-integration helpers
# ───────────────────────────────────────────────────────────────────────

def _integrate_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    Integrate `integrand_callable(s_1, ..., s_m, *free_ext_vals)` over
    the polytope `{s : a_int · s + c_eff > 0 for all constraints}`.

    s_constraints is a list of tuples `(a_int_list_of_len_m, c_eff)`.
    """
    if m == 0:
        # Zero integration variables — the "integrand" is just a number.
        # Still have to check the constraints (they may be vacuous or
        # infeasible).
        for (a_int, c_eff) in s_constraints:
            if c_eff <= 0:
                return 0.0 + 0.0j
        val = integrand_callable(*free_ext_vals)
        return complex(val)

    if m == 1:
        return _integrate_1d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    if m == 2:
        return _integrate_2d_polytope(
            integrand_callable, s_constraints, free_ext_vals
        )

    return _integrate_nd_polytope(
        integrand_callable, s_constraints, free_ext_vals, m
    )


def _complex_quad(integrand_callable, s_slot_index, other_args, lower, upper):
    """
    1D quadrature of `integrand_callable` along `s_slot_index`, with
    the other arguments fixed to `other_args`, over `[lower, upper]`.

    Real and imaginary parts are integrated separately via
    `scipy.integrate.quad`, which handles ±inf bounds natively.
    """
    from scipy.integrate import quad

    def _eval(s_val):
        args = list(other_args)
        args.insert(s_slot_index, float(s_val))
        val = integrand_callable(*args)
        return complex(val)

    def f_re(s_val):
        return _eval(s_val).real

    def f_im(s_val):
        return _eval(s_val).imag

    re_val, _ = quad(f_re, lower, upper, limit=200)
    try:
        im_val, _ = quad(f_im, lower, upper, limit=200)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _resolve_1d_bounds(s_constraints, s_index):
    """
    For the given integration-variable index `s_index`, intersect all
    half-line constraints `a_i s_{s_index} + (other terms already
    substituted) + c_eff > 0` into a single interval `[L, U]`.

    Each element of `s_constraints` is `(a_int_list, c_eff_scalar)`
    where `a_int_list` has length equal to the number of integration
    variables. For this 1D-on-one-axis pass we assume the other axes'
    coefficients are zero (i.e., the caller has already substituted
    them).

    Returns (L, U). If infeasible, L >= U and the caller should return
    0 without running the quadrature.
    """
    L, U = -math.inf, math.inf
    for (a_int, c_eff) in s_constraints:
        a = a_int[s_index]
        if abs(a) < 1e-15:
            if c_eff <= 0:
                return math.inf, -math.inf  # infeasible sentinel
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    return L, U


def _integrate_1d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """Single integration variable. The polytope is an interval."""
    L, U = _resolve_1d_bounds(s_constraints, s_index=0)
    if L >= U:
        return 0.0 + 0.0j
    return _complex_quad(
        integrand_callable,
        s_slot_index=0,
        other_args=list(free_ext_vals),
        lower=L,
        upper=U,
    )


def _integrate_2d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """
    Two integration variables s_0, s_1.

    Use `scipy.integrate.nquad` with `s_0` as the innermost integral
    and `s_1` as the outermost. The `s_0` bounds depend on s_1; the
    `s_1` bounds are extracted from constraints with zero coefficient
    on `s_0` (if any), defaulting to `(-inf, +inf)` if there are no
    pure-`s_1` constraints.
    """
    from scipy.integrate import nquad

    # Bounds on s_0 given s_1
    def bounds_s0(s_1_val):
        # Substitute s_1 into each constraint: a_0 s_0 + a_1 s_1 + c_eff > 0
        sub_constraints = []
        for (a_int, c_eff) in s_constraints:
            new_c_eff = c_eff + a_int[1] * s_1_val
            sub_constraints.append(([a_int[0], 0.0], new_c_eff))
        return _resolve_1d_bounds(sub_constraints, s_index=0)

    # Bounds on s_1: use constraints where a_0 = 0 (pure-s_1); else fall
    # back to (-inf, +inf) and rely on the inner bounds to keep things
    # finite.
    L1, U1 = math.inf, -math.inf
    pure_s1_found = False
    tmp_L, tmp_U = -math.inf, math.inf
    for (a_int, c_eff) in s_constraints:
        if abs(a_int[0]) < 1e-15:
            pure_s1_found = True
            a = a_int[1]
            if abs(a) < 1e-15:
                if c_eff <= 0:
                    return 0.0 + 0.0j
                continue
            bound = -c_eff / a
            if a > 0 and bound > tmp_L:
                tmp_L = bound
            elif a < 0 and bound < tmp_U:
                tmp_U = bound
    if pure_s1_found:
        L1, U1 = tmp_L, tmp_U
        if L1 >= U1:
            return 0.0 + 0.0j
    else:
        L1, U1 = -math.inf, math.inf

    def f_re(s_0, s_1):
        args = [float(s_0), float(s_1)] + list(free_ext_vals)
        return complex(integrand_callable(*args)).real

    def f_im(s_0, s_1):
        args = [float(s_0), float(s_1)] + list(free_ext_vals)
        return complex(integrand_callable(*args)).imag

    re_val, _ = nquad(f_re, [bounds_s0, (L1, U1)], opts={'limit': 200})
    try:
        im_val, _ = nquad(f_im, [bounds_s0, (L1, U1)], opts={'limit': 200})
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_nd_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    General `m >= 3` case — not exercised by the MVP. Raises so that
    the orchestrator can fall back to Phase I for diagrams that need
    it. Extension 1 will implement this via `scipy.integrate.nquad`
    with nested bound functions constructed from the constraint set.
    """
    raise NotImplementedError(
        f"Polytope integration with m = {m} integration variables is "
        "not implemented in the MVP (only m <= 2 is supported). "
        "Caller should fall back to Phase I for this kernel group."
    )


# ───────────────────────────────────────────────────────────────────────
# Edge → propagator-index lookup (matches _resolve_edge_propagator_data)
# ───────────────────────────────────────────────────────────────────────

def _lookup_prop_indices(typed_diagram, edge_key):
    """
    Look up (resp_row, phys_col) propagator indices for a prediagram
    edge, using the same fallback order as
    `_resolve_edge_propagator_data` in `msrjd/integration/symbolic.py`.
    """
    u, v, lbl = edge_key
    prop_indices = typed_diagram.propagator_indices
    td_edge_keys = list(typed_diagram.edge_types.keys())

    # 1. Exact match on (u, v, lbl)
    for td_ek in td_edge_keys:
        if td_ek == (u, v, lbl):
            return prop_indices[td_ek]

    # 2. Match on (u, v, None)
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v) and (
            len(td_ek) < 3 or td_ek[2] is None
        ):
            return prop_indices[td_ek]

    # 3. First edge matching (u, v) regardless of label
    for td_ek in td_edge_keys:
        if (td_ek[0], td_ek[1]) == (u, v):
            return prop_indices[td_ek]

    raise KeyError(
        f"No propagator indices found for edge ({u}, {v}, {lbl}) in "
        f"typed diagram."
    )

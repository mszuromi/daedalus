"""
msrjd.integration.time_domain.final_integral
=============================================
Vertex-time integration on a contracted parent diagram.

MVP scope
---------
Only tree-level (loop_number == 0) typed diagrams are handled in the
initial build. Tree-level is the right proving ground for the Phase J
evaluation layer because it exercises:

- time-domain retarded propagator lookup per edge,
- the global Heaviside convention (SageMath default, fixed pipeline-wide),
- vertex-time integration with explicit polyhedral bounds from
  Heaviside factors,
- global translation invariance / origin pinning,
- numerical dispatch,

**without** needing the kernel reduction / caching / contraction
machinery (Phases 3-5 of the hybrid pipeline). Loop cases are deferred
to Extension 1.

Integration strategy (MVP)
--------------------------
The MVP uses SageMath's generic symbolic `integrate()` over the
non-leaf vertex times. Each edge factor carries a `heaviside(t_head -
t_tail)` from `G_t_entry`, and Sage's symbolic integration is expected
to resolve the polyhedral region implicitly. This is provisional —
the mature engine (introduced in later extensions) will extract the
Heaviside inequalities explicitly, assemble the polytope, and do
closed-form exponential integration. See the plan file for details.

`sage.all.integrate()` may hang on nontrivial Heaviside-gated
integrands; all calls are wrapped in a SIGALRM timeout. If the timeout
fires, the function returns a structured failure dict so the
orchestrator can fall back to Phase I for the affected kernel group.
"""

import signal

from sage.all import SR, oo, integrate as sage_integrate

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
)


# ───────────────────────────────────────────────────────────────────────
# Symbolic integration with a hard wall-clock timeout
# ───────────────────────────────────────────────────────────────────────

class _IntegrationTimeout(Exception):
    pass


def _timed_integrate(expr, var, lower, upper, timeout_sec=30):
    """
    Run `sage.all.integrate(expr, var, lower, upper)` with a wall-clock
    timeout. Raises `_IntegrationTimeout` if Sage doesn't return in
    `timeout_sec` seconds.
    """
    def _handler(signum, frame):
        raise _IntegrationTimeout()

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_sec))
    try:
        return sage_integrate(expr, var, lower, upper)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


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
    timeout_sec=30,
):
    r"""
    Vertex-time integration for a TREE-LEVEL typed diagram.

    Builds the symbolic time-domain integrand

        combined_prefactor · ∏_{edges (u,v)} G^R_{phys, resp}(t_v - t_u)

    where `t_leaf = ext_time_vars[j]` for the j-th external leaf and
    `t_v` for each non-leaf vertex is a fresh SR integration variable.
    Then integrates out the non-leaf vertex times over `(-∞, +∞)` (the
    Heaviside factors in each `G_t_entry` cut the integration region
    down to the causal polyhedron).

    Parameters
    ----------
    typed_diagram : TypedDiagram
        Must have `loop_number == 0` (asserted below). Needed for the
        prediagram `D`, leaf list, and `propagator_indices`.
    representative_ir : dict
        Output of `build_integrand_stationary`. Used only for the
        loop_number sanity check — all graph topology comes from
        `typed_diagram`.
    propagator_data : dict
        Standard pipeline propagator dict with keys `'pole_vals'` and
        `'C_mats'`.
    combined_prefactor : SR
        Sum of scalar prefactors over diagrams in the kernel group.
    ext_time_vars : list of SR
        `k` external time variables, one per leaf (in `leaves` order).
    num_params : dict or None
        Numerical parameter substitutions for the propagator matrix.
        Passing them here reduces symbolic blow-up in the integrand
        (exponential coefficients become numbers instead of expressions).
    origin_leaf_idx : int
        Which external leaf's time to pin to zero (global translation
        invariance). Default 0. If set to None, no pinning is applied and
        the result remains a function of all `k` external times — it will
        still depend only on the `k-1` differences by stationarity.
    timeout_sec : int
        Wall-clock timeout for each call to Sage's `integrate()`.

    Returns
    -------
    dict with keys:
        'status' : 'ok' | 'timeout' | 'unevaluated'
        'contribution' : SR expression
            The evaluated tree-level contribution. For status != 'ok',
            contains whatever Sage managed to return (often an
            unevaluated `integrate(...)` SR expression).
        'integration_vars' : list of SR
            The non-leaf vertex time variables that were integrated out.
        'integrand' : SR
            The symbolic integrand (post-substitution) before integration
            — useful for debugging.
    """
    # Safety net against accidental misuse once loop kernels are added.
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

    # Build the numerical / symbolic time-domain propagator matrix. We
    # parameterize it in a single fresh time symbol `_t_td_` and then
    # substitute per-edge time differences via G_t_entry.
    t_sym = SR.var('_t_td_')
    G_t_matrix = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

    # Assign a time variable to every vertex.
    #   - leaves get their prescribed external time (possibly pinned)
    #   - non-leaf vertices get fresh integration variables s_<idx>
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

    # Build the product of edge time-domain propagators.
    integrand = SR(1)
    for idx, (u, v, lbl) in enumerate(D.edges()):
        # Look up the (resp_row, phys_col) indices for this edge via the
        # typed diagram's propagator_indices table. Fall back to matching
        # on (u, v, lbl) and then (u, v, None) if the key is shaped
        # differently (same fallback pattern as
        # `_resolve_edge_propagator_data` in symbolic.py).
        ri_pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        ri, pi = ri_pi

        dt = vertex_time[v] - vertex_time[u]
        edge_factor = G_t_entry(G_t_matrix, pi, ri, dt, include_heaviside=True)
        integrand = integrand * edge_factor

    # Apply the combined prefactor before integration — this keeps the
    # numerical magnitudes in a sane range during symbolic work.
    if combined_prefactor is not None:
        integrand = SR(combined_prefactor) * integrand

    # Integrate over the non-leaf vertex times. Heaviside factors do
    # double duty as the integration bounds; passing (-oo, +oo) lets
    # Sage collapse the region implicitly.
    current = integrand
    status = 'ok'

    for s_var in integration_vars:
        try:
            current = _timed_integrate(current, s_var, -oo, oo,
                                       timeout_sec=timeout_sec)
        except _IntegrationTimeout:
            status = 'timeout'
            break
        except Exception:
            # Any other Sage failure — leave `current` as-is and mark
            # unevaluated so the orchestrator can fall back to Phase I.
            status = 'unevaluated'
            break

    try:
        current = current.simplify_full()
    except Exception:
        try:
            current = current.simplify_rational()
        except Exception:
            pass

    return {
        'status': status,
        'contribution': current,
        'integration_vars': integration_vars,
        'integrand': integrand,
    }


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

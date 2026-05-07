"""
msrjd.integration.time_domain.final_integral
=============================================
Vertex-time integration on a tree-level diagram via explicit numerical
quadrature, with proper handling of the δ(t) component of any
"instantaneous" propagator entry.

MVP scope
---------
Only tree-level (loop_number == 0) typed diagrams are handled. Tree-level
is the right proving ground for the Phase J evaluation layer because it
exercises:

- time-domain retarded propagator lookup per edge,
- polytope extraction from explicit Heaviside factors,
- **δ-edge subset enumeration** (δ(t) components of instantaneous
  couplings like `ñ × δn` in MSR-JD),
- vertex-time integration over the retarded polytope,
- global translation invariance / origin pinning,
- numerical dispatch,

**without** needing the kernel reduction / caching / contraction
machinery. Loop cases are deferred to Extension 1.

Integration strategy
--------------------
The retarded propagator for an edge `(u -> v)` with matrix index
`(phys=p, resp=r)` decomposes into two pieces:

    G_R[p, r](t_v - t_u)
      =  delta_coeff[p, r] · δ(t_v - t_u)
       + Θ(t_v - t_u) · smooth[p, r](t_v - t_u)

The δ piece encodes an instantaneous response (nonzero whenever the
frequency-domain entry has a nonzero `ω → ∞` limit, which is the case
for any instantaneous coupling in the MSR-JD action). The smooth piece
is the usual pole-residue sum.

For a tree diagram with `|E|` edges, the full integrand is a product
of `|E|` such factors. Expanding the product yields `2^|E|` terms; we
enumerate them by picking a subset `S ⊆ edges` to take in its δ form
and the complement in its smooth form:

  Σ_{S ⊆ edges}  ∫ ds_1 … ds_m
                 · (∏_{e ∈ S} delta_coeff[e] · δ(t_{v_e} - t_{u_e}))
                 · (∏_{e ∉ S} Θ(t_{v_e} - t_{u_e}) · smooth[e](t_{v_e} - t_{u_e}))
                 · combined_prefactor

For each subset:

1. **Delta-edge equations** `t_{v_e} − t_{u_e} = 0` for `e ∈ S` are
   solved to eliminate integration variables by substitution. In the
   MVP star-tree case (single source vertex, all edges connecting source
   to leaves), one δ-edge pins the source time to a specific leaf time;
   two or more δ-edges force equality among several leaf times, which
   is a shot-noise δ(τ=0) contribution and is **skipped** for the
   continuous-callable return type.

2. **Smooth edges** contribute both a `fast_callable`-compiled factor
   (JIT'd over `CDF`) and a linear retardation constraint `t_v > t_u`
   that's added to the polytope.

3. The per-subset contribution is then a numerical quadrature over the
   reduced set of integration variables with the reduced polytope.

4. All subset contributions are summed into the final callable.

The public entry point `integrate_tree_diagram` returns a Python
callable `contribution(*ext_time_values) -> complex`.

Numerical overflow mitigation
-----------------------------
Each per-subset smooth product is `.expand()`'d into a sum of
single-exponential terms before JIT-compilation, so
`scipy.integrate.quad` cannot overflow IEEE doubles when sampling at
large negative `s`. (See the 2026-04-08 overflow fix in the CHANGELOG.)
"""

import math

from sage.all import SR, fast_callable, CDF, solve as sage_solve

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from msrjd.core.vertices import NoiseSourceType


# ───────────────────────────────────────────────────────────────────────
# Quadrature accuracy knob
# ───────────────────────────────────────────────────────────────────────
# Controls scipy.integrate.quad / nquad parameters for the vertex-time
# integrals. Loosen these for fast iterative checks; tighten for
# publication-quality results.
#
# Usage from a notebook cell (BEFORE running the numerics cell):
#   from msrjd.integration.time_domain import final_integral
#   final_integral.QUAD_OPTS = {'limit': 30, 'epsrel': 1e-3}
#
QUAD_OPTS = {
    'limit': 200,      # max subintervals for scipy.integrate.quad / nquad
}

# ───────────────────────────────────────────────────────────────────────
# Cumulant-kernel τ_v integration cap (non-local noise sources)
# ───────────────────────────────────────────────────────────────────────
# Diagrams with a NoiseSourceType vertex carry an extra integration
# variable τ_v parametrising the relative time between the source's
# legs (per-leg time map for non-local cumulant kernels).  The
# kernel itself decays on its natural timescale (e.g., σ for a
# Gaussian), so integrating τ_v over a half-infinite range
# (retard_L, +∞) — which is what the polytope alone gives —
# leaves scipy.quad's tan-substitution coordinate transform free
# to compress the kernel's central peak near the boundary, where
# adaptive sampling intermittently misses it.  Capping τ_v ∈
# (-CAP, +CAP) collapses the range to a finite interval where
# adaptive quadrature is well-behaved.  ±50 is safe for kernels
# with σ ≤ 5; loosen to ±200 (the polytope OUTER_CAP) if your
# kernel has heavy tails:
#   from msrjd.integration.time_domain import final_integral
#   final_integral.TAU_KERNEL_CAP = 200.0
TAU_KERNEL_CAP = 50.0


# ───────────────────────────────────────────────────────────────────────
# Tree-level vertex-time integration
# ───────────────────────────────────────────────────────────────────────

def _loop_number_from_graph(typed_diagram):
    """Compute loop number from the diagram's graph structure.

    For a connected graph: L = |E| - |V| + 1.
    This avoids any frequency-domain dependency.
    """
    D = typed_diagram.prediagram[0]
    return D.num_edges() - D.num_verts() + 1


def integrate_diagram(
    typed_diagram,
    propagator_data,
    combined_prefactor,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    external_fields=None,
    representative_ir=None,  # deprecated, kept for backward compat
):
    r"""
    Vertex-time integration for a typed Feynman diagram at ANY loop
    order, evaluated via explicit numerical quadrature.

    Previously this function was named ``integrate_tree_diagram`` and
    asserted tree-level.  The core algorithm -- assign a time to each
    internal vertex, integrate over those times, and enforce the
    retarded Heaviside on every edge -- works identically for loop
    diagrams because our enumeration produces DAGs (the "loop" in
    Feynman terminology is a topological cycle in the underlying
    undirected graph, not a cyclic directed path).  Multi-edges
    between the same vertex pair are already supported through the
    3-tuple ``(u, v, label)`` edge keys used by
    ``_lookup_prop_indices``.

    Returns a dict whose `contribution` value is a Python callable

        f(*ext_time_values) -> complex

    taking `k` positional arguments in **canonical** order: position i
    is the time of ``external_fields[i]``.  If `origin_leaf_idx` is
    not None, the value supplied at that position is ignored (it was
    pinned to zero during integrand construction).

    Parameters
    ----------
    typed_diagram : TypedDiagram
        The typed diagram.  Any loop order is accepted; the
        integrator treats every internal vertex as an integration
        variable and every edge as a retarded-propagator factor.
        Needed for the prediagram `D`, leaf list, and
        ``propagator_indices``.
    propagator_data : dict
        Must contain `'pole_vals'`, `'C_mats'`, and optionally
        `'D_delta'` (for delta-coefficient detection).
    combined_prefactor : SR or numeric
        Sum of scalar prefactors over diagrams in the kernel group.
    ext_time_vars : list of SR
        `k` external time variables in canonical order:
        ``ext_time_vars[i]`` is the time of ``external_fields[i]``.
    num_params : dict or None
        Numerical parameter substitutions for the propagator matrix AND
        the combined prefactor. Required if either the propagator
        entries or the prefactor contain free symbolic parameters — the
        JIT-compiled integrand cannot be built until every symbol
        except the integration variables and external times has been
        substituted.
    origin_leaf_idx : int or None
        Which canonical position to pin to zero (i.e., which entry of
        `external_fields` provides the base time). Default 0.
    external_fields : list of tuple or None
        The canonical external field list as specified by the user,
        e.g. ``[('dn',1), ('dn',1), ('dn',2)]``.  Used to map each
        leaf to its canonical position so that ``contribution(t_1,
        t_2, t_3)`` always has position i = time of
        ``external_fields[i]``, regardless of the diagram's internal
        leaf ordering.  If None, falls back to position-based mapping
        (leaf j → ext_time_vars[j]).

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
    loop_number = _loop_number_from_graph(typed_diagram)

    D = typed_diagram.prediagram[0]
    leaves = list(typed_diagram.prediagram[2])
    leaf_set = set(leaves)

    if len(ext_time_vars) != len(leaves):
        raise ValueError(
            f"ext_time_vars has length {len(ext_time_vars)} but "
            f"the diagram has {len(leaves)} leaves."
        )

    # ── 1. Numerical G(t) matrix (smooth + delta parts) ──────────
    t_sym = SR.var('_t_td_')
    G_t_obj = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

    # ── 2. Enumerate inter-vertex Wick contractions ───────────────
    # For correlators with repeated external field types (e.g. two
    # dn₁ legs at different spacetime points), each DISTINCT way to
    # assign canonical positions to leaves is a separate Wick
    # contraction that contributes to the connected correlator.
    #
    # The dedup in symmetry.py merges diagrams that differ only in
    # which same-type leaf connects to which vertex (they have the
    # same field-type multiset per vertex).  We compensate here by
    # summing over all such permutations.
    #
    # For same-vertex permutations (e.g. 2 dn₁ both at one vertex),
    # the integrand is invariant under swap (commutative product),
    # so summing 2! mappings gives 2x overcounting → divide by 2!.
    # For cross-vertex permutations (dn₁ at different vertices), the
    # integrands differ → summing gives the full answer.
    import itertools as _itertools
    from math import factorial as _factorial

    if external_fields is not None and len(external_fields) == len(leaves):
        _leaf_fields = [typed_diagram.external_legs.get(lf) for lf in leaves]
        # Group canonical positions and leaves by field type
        _cp_by_field = {}
        _leaves_by_field = {}
        for cp, field in enumerate(external_fields):
            _cp_by_field.setdefault(field, []).append(cp)
        for j, field in enumerate(_leaf_fields):
            _leaves_by_field.setdefault(field, []).append(j)

        # Enumerate all canonical-to-leaf mappings
        _all_mappings = [{}]
        for field in sorted(_cp_by_field.keys(), key=str):
            cps = _cp_by_field[field]
            lfs = _leaves_by_field.get(field, [])
            if len(cps) != len(lfs):
                _all_mappings = [{j: j for j in range(len(leaves))}]
                break
            perms = list(_itertools.permutations(lfs))
            new_mappings = []
            for m in _all_mappings:
                for perm in perms:
                    nm = dict(m)
                    for cp, lf_idx in zip(cps, perm):
                        nm[cp] = lf_idx
                    new_mappings.append(nm)
            _all_mappings = new_mappings

        # Compensation: for each internal vertex V, for each same-type
        # field group at V with n_V leaves, divide by n_V!.  This
        # removes overcounting from within-vertex permutations (which
        # give the same integrand by commutativity).
        _vertex_of_leaf = {}
        for ek in typed_diagram.edge_types:
            u, v = ek[0], ek[1]
            if u in leaf_set and v not in leaf_set:
                _vertex_of_leaf[u] = v
            elif v in leaf_set and u not in leaf_set:
                _vertex_of_leaf[v] = u
        _vertex_field_counts = {}
        for lf in leaves:
            v = _vertex_of_leaf.get(lf)
            if v is None:
                continue
            field = typed_diagram.external_legs.get(lf)
            _vertex_field_counts.setdefault(v, {}).setdefault(field, 0)
            _vertex_field_counts[v][field] += 1
        _compensation = 1
        for v, fcounts in _vertex_field_counts.items():
            for field, count in fcounts.items():
                _compensation *= _factorial(count)
    else:
        # SAFETY WARNING: falling back to identity leaf→position
        # mapping.  For diagrams whose leaves have MIXED field types
        # (e.g. one δn_1 and one δv_2), the enumeration's leaf order
        # is not guaranteed to match ``external_fields`` — a mismatch
        # here produces a τ → −τ mirror image of the physical
        # correlator.  Always pass ``external_fields`` when k ≥ 2.
        _leaf_field_list = [typed_diagram.external_legs.get(lf)
                            for lf in leaves]
        if len(set(_leaf_field_list)) > 1:
            import warnings as _warnings
            _warnings.warn(
                "integrate_tree_diagram: external_fields not provided "
                "for a diagram with mixed leaf field types "
                f"({_leaf_field_list}).  The canonical leaf→position "
                "mapping will fall back to identity, which may produce "
                "a τ → −τ mirror image of the physical correlator.  "
                "Pass external_fields to fix.",
                stacklevel=2,
            )
        _all_mappings = [{j: j for j in range(len(leaves))}]
        _compensation = 1

    # Build vertex_time for the FIRST mapping; subsequent mappings
    # handled by permuting the positional arguments in the wrapper.
    vertex_time = {}
    _first_mapping = _all_mappings[0]
    for cp, leaf_idx in _first_mapping.items():
        lf = leaves[leaf_idx]
        t_ext = ext_time_vars[cp]
        if origin_leaf_idx is not None and cp == origin_leaf_idx:
            t_ext = SR(0)
        vertex_time[lf] = t_ext

    internal_vertices = [v for v in D.vertices() if v not in leaf_set]
    integration_vars = []
    for v in internal_vertices:
        s_v = SR.var(f's_v{v}_td_', latex_name=rf's_{{v_{{{v}}}}}')
        vertex_time[v] = s_v
        integration_vars.append(s_v)

    # ── 2b. Non-local cumulant noise sources: per-leg time map ────
    # For each NoiseSourceType vertex, the response legs sit at
    # *independent* times coupled by a non-local kernel κ^{(n)}(τ).
    # We anchor the vertex at its existing time symbol (= leg-0
    # time) and introduce one extra integration variable τ per
    # noise source, with leg-1 time = anchor − τ.  Edges leaving
    # this vertex through leg 1 (matched by their resp_leg pop_idx)
    # are routed through the leg-1 time symbol.  Plain SourceType
    # vertices (cortical Poisson, GTaS auto-cumulant) keep the
    # single-time semantics — vertex_leg_time stays empty for them
    # and the existing edge-build code path is unaffected.
    vertex_leg_time = {}        # {v: {leg_pop_idx_0based: SR time}}
    noise_source_specs = {}     # {v: list of cumulant_specs dicts}
    extra_tau_syms = []         # list of (tau_sym, vertex) pairs
    vertex_assignments = (
        getattr(typed_diagram, 'vertex_assignments', None) or {}
    )
    for v, vtype in vertex_assignments.items():
        if not isinstance(vtype, NoiseSourceType):
            continue
        if not vtype.cumulant_specs:
            continue
        # All specs on this vertex share the same leg multiset
        # (they were grouped by extract_source_types).  Read the
        # 0-based leg ordering from the first spec.
        legs0 = vtype.cumulant_specs[0]['legs']    # e.g. (0, 1)
        anchor_leg = legs0[0]
        other_leg  = legs0[1] if len(legs0) > 1 else legs0[0]
        if anchor_leg == other_leg:
            # Auto-cumulant survived as non-local (kernel had no
            # delta, but legs are equal) — treat both legs at the
            # anchor time and leave τ-integration in.  Rare in
            # practice; keeps the code uniform.
            other_leg = anchor_leg
        # Anchor time = the existing internal-vertex symbol
        anchor_time = vertex_time[v]
        # τ symbol — distinct per noise vertex
        tau_sym = SR.var(
            f's_v{v}_tau_td_', latex_name=rf'\tau_{{v_{{{v}}}}}'
        )
        extra_tau_syms.append((tau_sym, v))
        integration_vars.append(tau_sym)
        # Per-leg time map: leg-0 = anchor; leg-1 = anchor − τ
        vertex_leg_time[v] = {
            anchor_leg: anchor_time,
            other_leg:  anchor_time - tau_sym,
        }
        noise_source_specs[v] = list(vtype.cumulant_specs)

    # ── 3. Gather per-edge info: ri, pi, dt, delta_coeff, smooth factor
    edges = list(D.edges())
    edge_info = []
    for (u, v, lbl) in edges:
        ri, pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        # Route edge tail through per-leg time when u is a non-local
        # noise source; otherwise use the standard single-time map.
        if u in vertex_leg_time:
            edge_resp_leg, _ = typed_diagram.edge_types[(u, v, lbl)]
            edge_pop_idx = edge_resp_leg[1] - 1  # 0-based
            t_u = vertex_leg_time[u].get(
                edge_pop_idx, vertex_time[u]
            )
        else:
            t_u = vertex_time[u]
        # Edge head: same logic (head can be a noise source if the
        # diagram has source-to-source edges, which doesn't arise
        # for current models; harmless in any case).
        if v in vertex_leg_time:
            edge_resp_leg_v, _ = typed_diagram.edge_types[(u, v, lbl)]
            edge_pop_idx_v = edge_resp_leg_v[1] - 1
            t_v = vertex_leg_time[v].get(
                edge_pop_idx_v, vertex_time[v]
            )
        else:
            t_v = vertex_time[v]
        dt = SR(t_v - t_u)
        delta_c = G_t_delta_coeff(G_t_obj, pi, ri)
        smooth_factor = G_t_entry(G_t_obj, pi, ri, dt, include_heaviside=False)
        edge_info.append({
            'u': u, 'v': v, 'lbl': lbl,
            'ri': ri, 'pi': pi,
            'dt_sym': dt,
            'delta_coeff': delta_c,
            'smooth_factor': smooth_factor,
        })

    # Combined prefactor (numerical)
    cp = SR(combined_prefactor) if combined_prefactor is not None else SR(1)
    if num_params:
        cp = cp.subs(num_params)

    # ── 3b. Non-local cumulant kernel substitution ────────────────
    # For each NoiseSourceType vertex, replace each placeholder
    # symbol ``z_kappa_<noise>_<order>_<i>_<j>`` in ``cp`` with the
    # actual kernel SR expression returned by the user's kernel_fn,
    # evaluated at the per-vertex τ integration symbol.  The signs
    # and combinatorial factors that ``_build_cumulant_action``
    # multiplied onto each placeholder (typically -1/2) are already
    # in ``cp``; the substitution carries them through.  The result
    # is a cp that is now an explicit function of the τ symbols,
    # which the existing fast_callable / nquad path handles
    # naturally because each τ is in ``integration_vars``.
    if noise_source_specs:
        kappa_subs = {}
        for v, specs in noise_source_specs.items():
            tau_sym_v = next(
                (ts for ts, vv in extra_tau_syms if vv == v), None
            )
            if tau_sym_v is None:
                continue
            for spec in specs:
                i_leg, j_leg = spec['legs'][0], spec['legs'][1]
                kappa_subs[spec['symbol']] = SR(
                    spec['kernel_fn'](i_leg, j_leg, tau_sym_v)
                )
        if kappa_subs:
            cp = cp.subs(kappa_subs)
            if num_params:
                # Re-substitute num_params now that kernel symbols
                # like ns.lambda_X, ns.mu_shift_diff have been
                # introduced via kernel_fn evaluation.
                cp = cp.subs(num_params)
            # Try to combine like terms (e.g. ordered pairs (i,j) and
            # (j,i) with symmetric kernels collapse to a single term).
            # simplify_full can be expensive but the cumulant prefactor
            # is a small SR expression so it's cheap here.
            try:
                cp = cp.simplify_full()
            except (ValueError, RuntimeError, AttributeError):
                pass

    # ── 4. External-time bookkeeping ─────────────────────────────
    free_ext_idx = [
        j for j in range(len(ext_time_vars))
        if (origin_leaf_idx is None or j != origin_leaf_idx)
    ]
    free_ext_syms = [ext_time_vars[j] for j in free_ext_idx]

    n_edges = len(edge_info)
    n_subsets_total = 2 ** n_edges

    # Expose the |S|=0 (all smooth) symbolic integrand and constraints
    # for debugging / display, matching the pre-fix return shape.
    display_stripped = cp
    for ei in edge_info:
        display_stripped = display_stripped * ei['smooth_factor']
    try:
        display_stripped = display_stripped.expand()
    except Exception:
        pass
    display_constraints = [ei['dt_sym'] for ei in edge_info]

    # Accumulators
    subset_contributions = []   # continuous smooth contributions (callable)
    delta_contributions = []    # shot-noise δ spikes (structured dicts)
    n_shotnoise_skipped = 0
    subset_diagnostics = []

    for subset_bits in range(n_subsets_total):
        delta_edges = [i for i in range(n_edges) if (subset_bits >> i) & 1]
        smooth_edges = [i for i in range(n_edges)
                        if not ((subset_bits >> i) & 1)]

        # Zero-delta-coeff edges contribute nothing when chosen as δ
        if any(abs(complex(edge_info[i]['delta_coeff'])) < 1e-15
               for i in delta_edges):
            continue

        # ── Solve the δ-edge equalities: eliminate integration vars
        # by substitution. For each δ edge, set dt_e = 0 and solve for
        # an integration variable appearing in the equation; if no
        # integration variable is available, the equation becomes a
        # constraint among external times (shot-noise δ, skip).
        substitutions = {}
        remaining_int_vars = list(integration_vars)
        ext_time_equalities = []  # residual constraints on ext times

        subset_infeasible = False
        for ei_idx in delta_edges:
            eq_expr = edge_info[ei_idx]['dt_sym'].subs(substitutions)
            eq_expr = SR(eq_expr)
            # Find an integration variable to solve for
            int_var_to_eliminate = None
            try:
                eq_vars = set(eq_expr.variables())
            except AttributeError:
                eq_vars = set()
            for iv in remaining_int_vars:
                if iv in eq_vars:
                    int_var_to_eliminate = iv
                    break
            if int_var_to_eliminate is not None:
                try:
                    sol = sage_solve(
                        eq_expr == 0, int_var_to_eliminate,
                        solution_dict=True,
                    )
                except Exception:
                    sol = []
                if not sol:
                    subset_infeasible = True
                    break
                substitutions[int_var_to_eliminate] = \
                    sol[0][int_var_to_eliminate]
                remaining_int_vars.remove(int_var_to_eliminate)
            else:
                # No integration variable to eliminate → this is a
                # constraint on external times alone. If it's
                # identically zero, the δ is satisfied trivially; if
                # not, it's a shot-noise δ(τ)-style contribution.
                ext_time_equalities.append(eq_expr)

        if subset_infeasible:
            continue

        # Shot-noise check: any nontrivial residual equality among
        # external times means this subset contributes a δ(τ) spike
        # at the hypersurface where that equality holds. Instead of
        # skipping it, compute a structured δ-contribution: a numeric
        # coefficient together with the linear equality and any
        # retardation half-space constraints, so downstream code can
        # insert it into a discrete τ grid.
        has_shotnoise = False
        nontrivial_equalities = []
        for eq in ext_time_equalities:
            try:
                if bool(eq.is_zero()):
                    continue
            except Exception:
                pass
            # Nontrivial equation → shot-noise
            has_shotnoise = True
            nontrivial_equalities.append(SR(eq))
        if has_shotnoise:
            n_shotnoise_skipped += 1
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'shotnoise',
                'ext_time_equalities': ext_time_equalities,
            })

            # For the MVP we only support the single-equality case
            # (one δ(a · τ + c) spike per subset). Multi-equality cases
            # would correspond to δ(τ_a − τ_b) · δ(τ_c − τ_d) style
            # "double-delta" spikes which are rare at tree level and
            # deferred to Extension 1.
            if len(nontrivial_equalities) != 1:
                continue

            # Build the numeric coefficient: combined_pf × ∏ δ-coeffs
            #                               × (smooth factors with δ subs applied)
            subset_factor_delta = cp
            for ei_idx in delta_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * SR(edge_info[ei_idx]['delta_coeff'])
                )
            for ei_idx in smooth_edges:
                subset_factor_delta = (
                    subset_factor_delta
                    * edge_info[ei_idx]['smooth_factor']
                )
            subset_factor_delta = subset_factor_delta.subs(substitutions)
            # NOTE: ``.subs(num_params)`` used to run here but it is
            # redundant: ``build_G_t_matrix(propagator_data, t_sym,
            # num_params=num_params)`` has already substituted
            # ``num_params`` into every ``edge_info[i]['smooth_factor']``
            # and ``edge_info[i]['delta_coeff']``, and ``cp`` was
            # substituted at the top of ``integrate_diagram``.  The only
            # remaining free variables here are ext-time symbols and
            # integration variables -- neither of which is in
            # ``num_params``.  Removed 2026-04-21 (audit Fix #A, ~6%
            # speedup on k=2 ell=1 quadratic Hawkes).
            try:
                subset_factor_delta = subset_factor_delta.expand()
            except Exception:
                pass

            # At the shot-noise hypersurface, any remaining integration
            # variables must already have been eliminated (otherwise it
            # wouldn't be a pure δ contribution — m_sub > 0 after all
            # deltas applied means we'd need to integrate further).
            # For the MVP star tree, this condition always holds.
            if remaining_int_vars:
                subset_diagnostics.append({
                    'status': 'shotnoise_with_remaining_int_vars',
                    'delta_edges': delta_edges,
                    'remaining': list(remaining_int_vars),
                })
                continue

            # Extract the linear form of the equality in terms of
            # free_ext_syms: a · x + c = 0.
            eq = nontrivial_equalities[0]
            try:
                eq_a = [float(eq.coefficient(s)) for s in free_ext_syms]
                eq_c = float(eq.subs({s: 0 for s in free_ext_syms}))
            except (TypeError, ValueError):
                continue

            # Check the equality is actually nontrivial (not all zero)
            if all(abs(a) < 1e-15 for a in eq_a):
                # Pure numeric residual: if nonzero, no contribution;
                # if zero, it's trivially satisfied (shouldn't happen
                # because we already filtered `is_zero()` above).
                continue

            # The symbolic factor evaluated at the δ surface is just
            # subset_factor_delta — it already has all the pin-substitutions
            # applied and can be evaluated numerically once free_ext_vals
            # are supplied. For the MVP case where remaining_int_vars is
            # empty, subset_factor_delta depends only on free_ext_syms
            # (or is a constant).
            try:
                coeff_free_vars = set(subset_factor_delta.variables())
            except AttributeError:
                coeff_free_vars = set()
            unexpected_in_coeff = coeff_free_vars - set(free_ext_syms)
            if unexpected_in_coeff:
                # Can't build a callable coefficient with free params left
                continue

            try:
                coeff_fc = fast_callable(
                    subset_factor_delta,
                    vars=list(free_ext_syms),
                    domain=CDF,
                )
            except Exception:
                continue

            # Retardation constraints at the δ point: for the MVP's
            # shot-noise subset (all edges δ, no smooth), there are
            # no retardation constraints. But in principle a mixed
            # δ+smooth shot-noise subset could have them.
            retard_data_delta = []
            for ei_idx in smooth_edges:
                c_retard = SR(
                    edge_info[ei_idx]['dt_sym']
                ).subs(substitutions)
                try:
                    a_ext = [
                        float(c_retard.coefficient(s))
                        for s in free_ext_syms
                    ]
                    c0 = float(c_retard.subs(
                        {s: 0 for s in free_ext_syms}
                    ))
                except (TypeError, ValueError):
                    continue
                retard_data_delta.append((a_ext, c0))

            delta_contributions.append({
                'coeff_fc': coeff_fc,
                'equality_a': eq_a,
                'equality_c': eq_c,
                'equality_symbolic': eq,
                'retardation_data': retard_data_delta,
                'delta_edges': list(delta_edges),
                'free_ext_idx': list(free_ext_idx),
            })
            continue

        # ── Build the stripped integrand for this subset
        subset_factor = cp
        for ei_idx in delta_edges:
            subset_factor = subset_factor * SR(edge_info[ei_idx]['delta_coeff'])
        for ei_idx in smooth_edges:
            subset_factor = subset_factor * edge_info[ei_idx]['smooth_factor']
        subset_factor = subset_factor.subs(substitutions)
        # NOTE: ``.subs(num_params)`` removed 2026-04-21 (audit Fix #A) --
        # redundant because every ingredient entering ``subset_factor``
        # is already num_params-substituted:
        #   * ``cp``                          -- subbed at line 351-352
        #   * ``edge_info[i]['smooth_factor']`` -- via build_G_t_matrix
        #                                       (called with num_params)
        #   * ``edge_info[i]['delta_coeff']`` -- via build_G_t_matrix
        # and ``.subs(substitutions)`` above only touches integration-
        # variable symbols.  Measured ~6% speedup on k=2 ell=1 quadratic
        # Hawkes (5.10s -> 4.80s across 7 1-loop diagrams).
        try:
            subset_factor = subset_factor.expand()
        except Exception:
            pass

        # Safe zero-check: `subset_factor == 0` triggers simplify_full()
        # which can hang or blow up Maxima for complex Hawkes integrands.
        # Instead, check structurally: a trivially-zero SR expression is
        # the SR integer 0 (caught via is_trivial_zero when available, or
        # by string comparison as a fallback).
        try:
            if subset_factor.is_trivial_zero():
                continue
        except AttributeError:
            if str(subset_factor) == '0':
                continue

        # Build retardation constraints for smooth edges (with δ subs applied)
        subset_retard = []
        for ei_idx in smooth_edges:
            c = SR(edge_info[ei_idx]['dt_sym']).subs(substitutions)
            subset_retard.append(c)

        m_sub = len(remaining_int_vars)
        fc_vars_sub = list(remaining_int_vars) + list(free_ext_syms)

        # Check that the stripped integrand only contains expected vars
        try:
            subset_free_vars = set(subset_factor.variables())
        except AttributeError:
            subset_free_vars = set()
        unexpected = subset_free_vars - set(fc_vars_sub)
        if unexpected:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': display_stripped,
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(subset_bits)}] stripped integrand "
                    f"contains unexpected free symbols {unexpected}; "
                    f"pass them via num_params."
                ),
            }

        # JIT-compile
        try:
            integrand_fc_sub = fast_callable(
                subset_factor, vars=fc_vars_sub, domain=CDF,
            )
        except Exception as exc:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': display_stripped,
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(subset_bits)}] fast_callable "
                    f"failed: {exc}"
                ),
            }

        # Extract linear coefficients for the polytope
        subset_constraint_data = []
        constraint_err = None
        for c in subset_retard:
            c_sr = SR(c)
            try:
                a_int = [float(c_sr.coefficient(v))
                         for v in remaining_int_vars]
                a_ext = [float(c_sr.coefficient(s)) for s in free_ext_syms]
                zero_subs = {v: 0
                             for v in list(remaining_int_vars)
                             + list(free_ext_syms)}
                c0 = float(c_sr.subs(zero_subs))
            except (TypeError, ValueError) as exc:
                constraint_err = exc
                break
            subset_constraint_data.append((a_int, a_ext, c0))

        # ── Cap each surviving τ_v integration variable to a finite
        # range ──────────────────────────────────────────────────────
        # Cumulant kernels (Gaussian, etc.) decay rapidly on a
        # kernel-natural timescale.  Without a finite cap, scipy.quad
        # integrates over (retard_L, +∞), and the adaptive Cauchy /
        # tan-substitution coordinate transform compresses the
        # kernel's central peak near the upper boundary of the
        # transformed parameter range.  The peak then gets
        # intermittently missed by the sampling — producing the
        # spurious spikes seen in non-local diagram contributions
        # at τ values where the external time t puts the kernel
        # peak in the "danger zone."  Capping τ_v ∈ (−CAP, +CAP)
        # collapses the integration to a finite interval where
        # adaptive quad is well-behaved.  Outside ±5σ a Gaussian
        # is < 1e-6 of its peak; ±50 with σ ~ 1 is overkill and
        # safe for any kernel with σ < 10.
        if extra_tau_syms and remaining_int_vars:
            n_iv = len(remaining_int_vars)
            n_ext = len(free_ext_syms)
            for (tau_s, _v) in extra_tau_syms:
                if tau_s not in remaining_int_vars:
                    continue
                idx = remaining_int_vars.index(tau_s)
                # Upper cap:  -τ_v + CAP > 0  ⇒  τ_v < CAP
                a_up  = [0.0] * n_iv
                a_up[idx] = -1.0
                subset_constraint_data.append(
                    (a_up, [0.0] * n_ext, TAU_KERNEL_CAP)
                )
                # Lower cap:  +τ_v + CAP > 0  ⇒  τ_v > -CAP
                a_lo  = [0.0] * n_iv
                a_lo[idx] = +1.0
                subset_constraint_data.append(
                    (a_lo, [0.0] * n_ext, TAU_KERNEL_CAP)
                )
        if constraint_err is not None:
            return {
                'status': 'failed',
                'contribution': None,
                'integration_vars': integration_vars,
                'stripped_integrand': display_stripped,
                'constraints': display_constraints,
                'reason': (
                    f"[subset {bin(subset_bits)}] constraint not "
                    f"linear: {constraint_err}"
                ),
            }

        # Fix E (2026-04-21): try the direct numerical per-edge
        # evaluator before settling for ``fast_callable(subset_factor
        # .expand())``.  The evaluator reconstructs
        #
        #   P · Π_e  Σ_k  C_e^{(k)} · exp(I · p_k · Δt_e)
        #
        # from the propagator's pole / residue data plus the already-
        # extracted ``subset_constraint_data`` (which carries each
        # smooth edge's Δt linear coefficients), without ever
        # materialising the distributed |edges|^|poles|-term sum that
        # fast_callable has to compile and re-walk on every scipy
        # sample.  Overflow-safe because each edge factor is bounded
        # by Σ_k |C_e^{(k)}| for Δt_e ≥ 0 (the Heaviside filter
        # guarantees that precondition -- see
        # ``_make_heaviside_filtered_integrand``).
        #
        # Falls back to ``integrand_fc_sub`` if the numerical
        # extraction fails for any reason (non-numerical prefactor
        # left in ``cp``, missing pole data, residue conversion
        # failure, etc.).  The fast_callable path is still built
        # above so the zero-check and free-variable-check have run;
        # this branch only replaces the *hot-path* evaluator.
        prefactor_num = cp
        for _ei_idx in delta_edges:
            prefactor_num = prefactor_num * SR(
                edge_info[_ei_idx]['delta_coeff']
            )
        smooth_edges_ri_pi = [
            (edge_info[_ei_idx]['ri'], edge_info[_ei_idx]['pi'])
            for _ei_idx in smooth_edges
        ]
        if vertex_leg_time:
            # Non-local cumulant kernel(s) are present in this
            # diagram (NoiseSourceType vertex with smooth κ kernel).
            # The fast pole/residue evaluator assumes the integrand
            # factorises as P · Π_e Σ_k C_e^{(k)} exp(i p_k Δt_e),
            # which doesn't hold once a Gaussian (or other
            # non-rational) kernel is in play.  Fall through to the
            # generic fast_callable path.  Plain (cortical / GTaS
            # auto) diagrams keep the fast evaluator.
            _fast_eval = None
        else:
            _fast_eval = _build_fast_subset_evaluator(
                propagator_data,
                prefactor_num,
                smooth_edges_ri_pi,
                subset_constraint_data,
                m_sub,
            )
        integrand_for_quad = (
            _fast_eval if _fast_eval is not None else integrand_fc_sub
        )

        # Build this subset's contribution callable
        def _make_subset_contrib(fc, cdata, m_val):
            def _contrib(free_vals):
                resolved = []
                for (a_int, a_ext, c0) in cdata:
                    c_eff = c0 + sum(a_ext[i] * free_vals[i]
                                     for i in range(len(a_ext)))
                    resolved.append((list(a_int), c_eff))
                return _integrate_polytope(fc, resolved, free_vals, m_val)
            return _contrib

        subset_contributions.append(
            _make_subset_contrib(
                integrand_for_quad, subset_constraint_data, m_sub,
            )
        )
        subset_diagnostics.append({
            'delta_edges': delta_edges,
            'smooth_edges': smooth_edges,
            'status': 'evaluated',
            'm_after_delta': m_sub,
            'evaluator': 'fast_numpy' if _fast_eval is not None else 'fast_callable',
        })

    # ── Build the final contribution callable ─────────────────────
    # The subset_contributions were built using the FIRST mapping.
    # To include all inter-vertex Wick contractions, we evaluate the
    # same integrand with the input arguments permuted for each
    # alternative mapping, then sum and divide by the compensation
    # factor (which removes overcounting of same-vertex permutations).
    _m0 = _all_mappings[0]
    _m0_inv = {v: k for k, v in _m0.items()}
    _k = len(ext_time_vars)

    # For each alternative mapping m, compute the permutation that
    # converts canonical-order inputs into the order the first-mapping
    # integrand expects.
    _perms = []
    for _m in _all_mappings:
        _perm = [0] * _k
        for _cp in range(_k):
            _perm[_m0_inv[_m[_cp]]] = _cp
        _perms.append(tuple(_perm))

    _comp = _compensation

    def contribution(*ext_time_values):
        if len(ext_time_values) != len(ext_time_vars):
            raise ValueError(
                f"contribution() expects {len(ext_time_vars)} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        total = 0.0 + 0.0j
        for perm in _perms:
            permuted = [ext_time_values[perm[j]] for j in range(_k)]
            free_vals = [float(permuted[j]) for j in free_ext_idx]
            for cfn in subset_contributions:
                total = total + complex(cfn(free_vals))
        return total / _comp

    # If non-local cumulant kernels were substituted in cp, expose
    # the τ-dependent prefactor (= the substituted, simplified cp,
    # already num_params-substituted) so the display layer can place
    # it inside the integral with the propagator factors.  When no
    # NoiseSourceType vertex is present, this is just cp (==
    # combined_prefactor.subs(num_params)) and the display layer
    # treats it as a τ-independent prefactor outside the integral.
    has_cumulant_kernel = bool(noise_source_specs)
    return {
        'status': 'ok',
        'contribution': contribution,
        'delta_contributions': delta_contributions,
        'integration_vars': integration_vars,
        'stripped_integrand': display_stripped,
        'constraints': display_constraints,
        'edge_info': edge_info,
        'n_subsets_evaluated': len(subset_contributions),
        'n_delta_contributions': len(delta_contributions),
        'n_shotnoise_skipped': n_shotnoise_skipped,
        'subset_diagnostics': subset_diagnostics,
        'cumulant_prefactor':       cp if has_cumulant_kernel else None,
        'has_cumulant_kernel':      has_cumulant_kernel,
    }


def eval_delta_contributions_on_tau_grid(
    delta_contributions,
    tau_grid,
    free_ext_dim=1,
    vary_index=0,
    fixed_values=None,
):
    r"""
    Convert a list of symbolic δ-spike contributions (from
    `integrate_tree_diagram` or `compute_correction_td`) into a
    discretized contribution on a 1D τ grid, optionally restricted to
    a slice of a higher-dimensional external-time space.

    Each delta contribution stores a linear equality
    `a · x + c = 0` in the free-external-time vector `x` (of length
    `free_ext_dim`). On a 1D slice where all-but-one entry of `x` is
    fixed, the equality collapses to `a_vary · τ + c' = 0` where
    `c' = c + Σ_{j ≠ vary_index} a_j · fixed_values[j]`. We solve
    `τ_fire = −c' / a_vary`, evaluate the coefficient callable at the
    (fixed + varying) point, check any retardation half-spaces, and
    deposit `coeff / |a_vary| / Δτ` into the nearest bin on
    `tau_grid`.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `integrate_tree_diagram` under
        `delta_contributions` or `compute_correction_td` under the
        same key. Each entry has `equality_a`, `equality_c`,
        `coeff_fc`, and `retardation_data`.
    tau_grid : 1-D numpy array
        Uniformly spaced grid of τ values along the axis being varied.
        Bin width is inferred as `tau_grid[1] - tau_grid[0]`.
    free_ext_dim : int, default 1
        Number of free-external-time dimensions that each
        `delta_contribution['equality_a']` is parameterized over.
        For k=2 with one leaf pinned this is 1; for k=3 it is 2.
    vary_index : int, default 0
        Which component of the free-external-time vector is swept by
        `tau_grid`. All other components are pinned to the
        corresponding value in `fixed_values`.
    fixed_values : dict or None
        Mapping `{j: value}` for indices `j ≠ vary_index`. Any index
        not supplied is pinned to 0.0. For the common k=3 "slice
        through the origin" the default (all zeros) is usually what
        the caller wants.

    Returns
    -------
    numpy.ndarray (complex)
        An array the same length as `tau_grid`, with zeros everywhere
        except at the bins where δ contributions fire.

    Notes
    -----
    A δ contribution whose equality is IDENTICALLY zero on the chosen
    slice (i.e., `a_vary == 0` but the remaining slice residual is
    also zero) corresponds to a "δ along the whole slice" — the
    contribution is continuous along the varying axis rather than
    concentrated at a single bin. These contributions are **skipped**
    by this helper with a silent pass; they should be handled by a
    full 2D grid evaluator or by adding an explicit continuous
    contribution to the smooth total. Callers can inspect
    `delta_contributions` directly to detect this case.
    """
    import numpy as np

    if free_ext_dim < 1:
        raise ValueError(f"free_ext_dim must be >= 1, got {free_ext_dim}")
    if not (0 <= vary_index < free_ext_dim):
        raise ValueError(
            f"vary_index={vary_index} out of range for "
            f"free_ext_dim={free_ext_dim}"
        )
    if fixed_values is None:
        fixed_values = {}

    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid.size < 2:
        raise ValueError("tau_grid must have at least 2 points")
    dtau = float(tau_grid[1] - tau_grid[0])
    out = np.zeros_like(tau_grid, dtype=complex)

    # Build the full fixed-values vector (length free_ext_dim, with
    # vary_index slot filled in per-evaluation).
    other_indices = [j for j in range(free_ext_dim) if j != vary_index]
    fixed_vec_template = [0.0] * free_ext_dim
    for j in other_indices:
        fixed_vec_template[j] = float(fixed_values.get(j, 0.0))

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            # Dimension mismatch between the delta contribution and
            # the caller's advertised free_ext_dim. Silently skip.
            continue

        a_vary = eq_a[vary_index]
        c_eff = eq_c + sum(
            eq_a[j] * fixed_vec_template[j] for j in other_indices
        )

        if abs(a_vary) < 1e-15:
            if abs(c_eff) > 1e-12:
                # Infeasible: the δ surface doesn't intersect this
                # slice at all → zero contribution.
                continue
            # DEGENERATE: the δ equality is satisfied EVERYWHERE on
            # this slice. The contribution is NOT a spike — it's a
            # smooth continuous function along the varying axis:
            #   C_degenerate(τ) = coeff_fc(τ) (no 1/dtau divisor)
            # This is the "pair-driven" piece: e.g. for two identical
            # pop-1 fields at the same time (δ(τ₁) on the τ₁=0
            # slice), the remaining smooth propagator to pop-2 gives
            # a decaying function of τ₂.
            for i_tau, tau_val in enumerate(tau_grid):
                eval_vec = list(fixed_vec_template)
                eval_vec[vary_index] = float(tau_val)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue
                # Check retardation constraints at this point
                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False
                        break
                    r_val = c0 + sum(
                        a_list[j] * eval_vec[j]
                        for j in range(free_ext_dim)
                    )
                    if r_val <= 0:
                        retard_ok = False
                        break
                if retard_ok:
                    out[i_tau] = out[i_tau] + val
            continue
        tau_fire = -c_eff / a_vary

        # Build the full point in free-ext-time space for evaluating
        # the coefficient callable
        eval_vec = list(fixed_vec_template)
        eval_vec[vary_index] = float(tau_fire)

        coeff_fc = dc['coeff_fc']
        try:
            coeff_val = complex(coeff_fc(*eval_vec))
        except Exception:
            continue

        # Check retardation constraints at the fire point
        retard_ok = True
        for (a_list, c0) in dc.get('retardation_data', []):
            if len(a_list) != free_ext_dim:
                retard_ok = False
                break
            val = c0 + sum(a_list[j] * eval_vec[j] for j in range(free_ext_dim))
            if val <= 0:
                retard_ok = False
                break
        if not retard_ok:
            continue

        # δ(a_vary · τ + c_eff) = δ(τ − τ_fire) / |a_vary|
        weight = coeff_val / abs(a_vary)

        # Find the nearest grid bin and add weight / dtau
        idx = int(np.argmin(np.abs(tau_grid - tau_fire)))
        out[idx] = out[idx] + weight / dtau

    return out


def eval_delta_contributions_on_2d_grid(
    delta_contributions,
    tau1_grid,
    tau2_grid,
    free_ext_dim=3,
    grid_axes=(1, 2),
    fixed_values=None,
):
    r"""
    Discretize δ-spike contributions onto a 2D grid for heatmap display.

    For each delta contribution with linear equality `a · x + c = 0`
    in the free-external-time vector `x`, this helper finds all 2D
    grid points where the equality is approximately satisfied (within
    half a grid step) and deposits the coefficient value there.

    Parameters
    ----------
    delta_contributions : list of dict
        As returned by `compute_correction_td` under `delta_contributions`.
    tau1_grid, tau2_grid : 1-D numpy arrays
        The two axes of the 2D heatmap grid. Must be uniformly spaced.
    free_ext_dim : int
        Total number of free external-time dimensions.
    grid_axes : tuple of two ints
        Which components of the free-ext vector correspond to the two
        grid axes. Default (1, 2) for k=3 with t_a=0 pinned.
    fixed_values : dict or None
        Values for any free-ext components NOT on the grid axes.
        Default: {0: 0.0} (the reference field at time 0).

    Returns
    -------
    numpy.ndarray (complex), shape (len(tau1_grid), len(tau2_grid))
        The discretized delta contributions. Add to the smooth `total_C`
        heatmap to get the full theory prediction.
    """
    import numpy as np

    if fixed_values is None:
        fixed_values = {0: 0.0}

    tau1 = np.asarray(tau1_grid, dtype=float)
    tau2 = np.asarray(tau2_grid, dtype=float)
    dt1 = float(tau1[1] - tau1[0]) if len(tau1) > 1 else 1.0
    dt2 = float(tau2[1] - tau2[0]) if len(tau2) > 1 else 1.0
    ax1, ax2 = grid_axes

    out = np.zeros((len(tau1), len(tau2)), dtype=complex)

    for dc in delta_contributions:
        eq_a = dc['equality_a']
        eq_c = dc['equality_c']
        coeff_fc = dc['coeff_fc']
        if len(eq_a) != free_ext_dim:
            continue

        a1 = eq_a[ax1]  # coefficient on the first grid axis
        a2 = eq_a[ax2]  # coefficient on the second grid axis

        # Contribution from fixed (non-grid) axes to the equality
        c_fixed = eq_c
        for idx in range(free_ext_dim):
            if idx != ax1 and idx != ax2:
                c_fixed += eq_a[idx] * float(fixed_values.get(idx, 0.0))

        # The equality on the grid is: a1·τ₁ + a2·τ₂ + c_fixed = 0
        # This is a line in the (τ₁, τ₂) plane. We find grid bins
        # that this line passes through.

        if abs(a1) < 1e-15 and abs(a2) < 1e-15:
            # No dependence on grid axes — either always or never fires
            if abs(c_fixed) < 1e-12:
                # Fires everywhere — evaluate coeff at every point
                for i, t1 in enumerate(tau1):
                    for j, t2 in enumerate(tau2):
                        eval_vec = [0.0] * free_ext_dim
                        for k_idx, v in fixed_values.items():
                            eval_vec[k_idx] = float(v)
                        eval_vec[ax1] = float(t1)
                        eval_vec[ax2] = float(t2)
                        try:
                            val = complex(coeff_fc(*eval_vec))
                        except Exception:
                            continue
                        # Check retardation
                        retard_ok = True
                        for (a_list, c0) in dc.get('retardation_data', []):
                            if len(a_list) != free_ext_dim:
                                retard_ok = False; break
                            rv = c0 + sum(a_list[m] * eval_vec[m]
                                          for m in range(free_ext_dim))
                            if rv <= 0:
                                retard_ok = False; break
                        if retard_ok:
                            # This is a 2D degenerate — the delta fires
                            # on the full 2D grid. No 1/dt divisor
                            # (the delta is in a direction orthogonal
                            # to the grid plane).
                            out[i, j] += val
            continue

        # The delta line a1·τ₁ + a2·τ₂ + c_fixed = 0 crosses the grid.
        # For each row i (fixed τ₁), solve for τ₂:
        #   τ₂_fire = -(a1·τ₁[i] + c_fixed) / a2
        # OR for each column j (fixed τ₂), solve for τ₁:
        #   τ₁_fire = -(a2·τ₂[j] + c_fixed) / a1
        # Use whichever axis has the larger coefficient (better resolved).

        if abs(a2) >= abs(a1):
            # Sweep along τ₁ axis, solve for τ₂ at each row
            for i, t1 in enumerate(tau1):
                t2_fire = -(a1 * t1 + c_fixed) / a2
                j = int(np.argmin(np.abs(tau2 - t2_fire)))
                if abs(tau2[j] - t2_fire) > dt2:
                    continue  # not within a grid cell

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1)
                eval_vec[ax2] = float(t2_fire)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                # The delta δ(a2·τ₂ + ...) has Jacobian 1/|a2|.
                # On the grid with spacing dt2, the bin density is
                # coeff / (|a2| · dt2).
                weight = val / (abs(a2) * dt2)
                out[i, j] += weight
        else:
            # Sweep along τ₂ axis, solve for τ₁ at each column
            for j, t2 in enumerate(tau2):
                t1_fire = -(a2 * t2 + c_fixed) / a1
                i = int(np.argmin(np.abs(tau1 - t1_fire)))
                if abs(tau1[i] - t1_fire) > dt1:
                    continue

                eval_vec = [0.0] * free_ext_dim
                for k_idx, v in fixed_values.items():
                    eval_vec[k_idx] = float(v)
                eval_vec[ax1] = float(t1_fire)
                eval_vec[ax2] = float(t2)
                try:
                    val = complex(coeff_fc(*eval_vec))
                except Exception:
                    continue

                retard_ok = True
                for (a_list, c0) in dc.get('retardation_data', []):
                    if len(a_list) != free_ext_dim:
                        retard_ok = False; break
                    rv = c0 + sum(a_list[m] * eval_vec[m]
                                  for m in range(free_ext_dim))
                    if rv <= 0:
                        retard_ok = False; break
                if not retard_ok:
                    continue

                weight = val / (abs(a1) * dt1)
                out[i, j] += weight

    return out


# ───────────────────────────────────────────────────────────────────────
# Fast numerical subset-integrand evaluator (Fix E, 2026-04-21)
# ───────────────────────────────────────────────────────────────────────

def _build_fast_subset_evaluator(
    propagator_data,
    prefactor_num,
    smooth_edges_ri_pi,
    subset_constraint_data,
    m_sub,
):
    r"""Return a Python callable that evaluates a subset's smooth
    integrand numerically, without going through
    ``fast_callable(subset_factor.expand())``.

    The subset integrand is structurally

        P · Π_e  Σ_k  C_e^{(k)} · exp(I · p_k · Δt_e)

    where ``P`` is the product of the combined prefactor and all
    δ-edge coefficients, the outer product runs over smooth edges,
    and the inner sum runs over poles.  ``fast_callable``'s overflow-
    safe form (Fix from 2026-04-08, commit 388fa7c) was to first call
    ``subset_factor.expand()``, distributing the ``|edges|^|poles|``
    cross-product into a sum of single exponentials.  For a 5-edge
    diagram with 4 poles that's ~988 single-exp terms, all compiled
    into the JIT tree.  cProfile of the k=2 ell=1 quadratic Hawkes
    V=5 diagram (2026-04-21) shows ~18 µs per ``fast_callable`` call,
    × 750 k samples per τ point, = the full 13 s of Phase J wall time.

    This evaluator skips the expansion entirely: each edge contributes
    ``Σ_k C_e^{(k)} exp(I p_k Δt_e)`` computed independently and
    multiplied in.  The product is bounded term-by-term by ``|C_e^{(k)}|``
    for Δt_e ≥ 0 (which the Heaviside filter guarantees -- see
    ``_make_heaviside_filtered_integrand``), so the pre-cancellation
    overflow that motivated the expand fix cannot occur.

    Returns None if the numerical extraction fails (e.g., ``prefactor``
    isn't purely numerical after ``num_params`` subs, or the propagator
    data lacks ``pole_vals`` / ``C_mats``).  Callers should fall back
    to ``fast_callable(subset_factor.expand())`` in that case.
    """
    import cmath as _cmath

    # ── Prefactor → complex scalar ──
    try:
        pref_c = complex(CDF(SR(prefactor_num)))
    except Exception:
        return None

    # ── Pole list (shared across edges) ──
    pole_vals = propagator_data.get('pole_vals')
    C_mats = propagator_data.get('C_mats')
    if pole_vals is None or C_mats is None:
        return None
    try:
        poles_tuple = tuple(complex(CDF(SR(p))) for p in pole_vals)
    except Exception:
        return None
    n_poles = len(poles_tuple)

    # ── Per-edge (residues, c0, int_pairs, ext_pairs) ──
    edge_data = []
    for (ri, pi), (a_int, a_ext, c0) in zip(
        smooth_edges_ri_pi, subset_constraint_data
    ):
        try:
            residues = tuple(
                complex(CDF(SR(C_mats[k][pi, ri])))
                for k in range(n_poles)
            )
        except Exception:
            return None
        # Sparse (position, coef) pairs — retardation ``Δt`` vectors
        # have exactly 1–2 nonzero entries regardless of m_sub.
        int_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_int)
            if abs(float(a)) > 1e-15
        )
        ext_pairs = tuple(
            (i, float(a)) for i, a in enumerate(a_ext)
            if abs(float(a)) > 1e-15
        )
        edge_data.append(
            (poles_tuple, residues, float(c0), int_pairs, ext_pairs)
        )
    edge_data_t = tuple(edge_data)
    m_offset = m_sub          # index into args where external times begin
    _cexp = _cmath.exp

    def evaluator(*args):
        # args = (s_0, ..., s_{m_sub-1}, t_free_0, t_free_1, ...)
        result = pref_c
        for (poles, residues, c0, int_pairs, ext_pairs) in edge_data_t:
            dt = c0
            for (i, a) in int_pairs:
                dt += a * args[i]
            for (i, a) in ext_pairs:
                dt += a * args[m_offset + i]
            # Σ_k r_k · exp(i·p_k·dt)
            edge_val = 0.0 + 0.0j
            for (p, r) in zip(poles, residues):
                edge_val += r * _cexp(1j * p * dt)
            result *= edge_val
        return result

    return evaluator


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
        # infeasible).  Θ(0) = 0 convention: boundary c_eff = 0 is OUTSIDE
        # the feasible region (the half-space is strictly open at Δt = 0).
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


def _make_heaviside_filtered_integrand(integrand_callable, s_constraints,
                                        free_ext_vals, m):
    r"""
    Wrap `integrand_callable` with an explicit Heaviside-product check.

    The polytope bounds we pass to `scipy.nquad` are only an approximation
    of the true polytope when some constraints couple multiple integration
    axes: cross-axis constraints must be deferred to an inner axis, and
    when the bounds function for an outer axis loses its lower or upper
    bound from such deferred constraints, we fall back to ±OUTER_CAP.
    That fallback admits regions geometrically OUTSIDE the true polytope.

    Physically the retarded propagator `G^R(Δt) = Θ(Δt) · G^sm(Δt)`
    vanishes on those regions via the Heaviside.  But our JIT-compiled
    integrand contains ONLY `G^sm`, never `Θ` — it relies entirely on the
    polytope bounds for retardation.  When the bounds overshoot, the
    integrand evaluates `G^sm` on a region where it should be zero, and
    for retarded poles (Im(ω) > 0) `G^sm(Δt) = C · exp(-γ Δt)` GROWS for
    `Δt < 0`, producing a spurious positive contribution.

    This wrapper explicitly multiplies by the Heaviside product, i.e.
    returns 0.0 whenever any `a_int · s + c_eff < 0`.  The polytope
    bounds then serve only as an optimization: they tighten the
    quadrature domain for speed, but correctness no longer depends on
    them being exact.

    Parameters
    ----------
    integrand_callable : callable
        The JIT-compiled smooth integrand `f(s_0, ..., s_{m-1},
        *free_ext_vals)`.
    s_constraints : list of (a_int, c_eff)
        Polytope constraints `a_int · s + c_eff > 0` for each retarded
        edge still active at this subset (strict inequality per Θ(0) = 0
        convention — the boundary Δt = 0 is OUTSIDE the feasible region).
        `c_eff` already has the current external-time values substituted
        in (via the caller).
    free_ext_vals : list
        Free external time values, passed through to `integrand_callable`.
    m : int
        Number of integration axes (`s_0, ..., s_{m-1}`).

    Returns
    -------
    callable f(*s_vals) → complex, with the Heaviside filter applied.
    """
    # Convention: Θ(0) = 0.  A constraint `a_int · s + c_eff > 0` is
    # STRICTLY required: the boundary `Δt = 0` is excluded.  Use `dt <= 0`
    # to kill both the exterior (strictly infeasible) AND the boundary.
    #
    # Fix D (2026-04-21): pre-extract constraints in SPARSE form so the
    # hot loop skips zero-coefficient axes entirely.  For retarded
    # propagators each constraint is `t_v − t_u > 0` which has exactly
    # two nonzero entries in `a_int` (regardless of `m`), so this
    # collapses an inner `for j in range(m)` loop into 2 iterations.
    #
    # A pre-check here handles constraints that are purely trivial
    # (all `a` zero): if `c_eff > 0` the constraint is always satisfied
    # and we drop it; if `c_eff <= 0` the polytope is empty and the
    # filter always returns 0.
    sparse = []
    always_empty = False
    for (a_int, c_eff) in s_constraints:
        c_eff_f = float(c_eff)
        pairs = tuple((j, float(a)) for j, a in enumerate(a_int)
                      if abs(float(a)) > 1e-15)
        if not pairs:
            # Pure constant constraint.  Θ(0) = 0: strict c_eff > 0.
            if c_eff_f <= 0.0:
                always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        sparse.append((c_eff_f, pairs))
    # Tuple-of-tuples for slightly faster iteration than list-of-tuples
    # (CPython's FOR_ITER has a specialized path for tuples).
    sparse_constraints = tuple(sparse)
    free_ext_tuple = tuple(free_ext_vals)

    if always_empty:
        def filtered_empty(*s_vals):
            return 0.0 + 0.0j
        return filtered_empty

    # Capture `free_ext_vals` as a LIST (not tuple) in the closure so
    # the argument-packing style matches the pre-Fix D code exactly —
    # `integrand_callable(*args)` with `args = list(s_vals) +
    # free_ext_list`.  Measured: `integrand_callable(*s_vals,
    # *free_ext_tuple)` (Python 3.5+ multi-unpack) triggers a slow
    # path inside Sage's fast_callable on the 1-loop m=3 workload
    # (~2× wall-clock regression vs. baseline) for reasons not yet
    # root-caused.  Empirically the single-unpack form matches
    # baseline performance while still benefiting from the sparse-
    # scan speedup below.
    free_ext_list = list(free_ext_vals)

    def filtered(*s_vals):
        # Heaviside check: every constraint must be > 0 (Θ(0) = 0).
        for c_eff, nzs in sparse_constraints:
            dt = c_eff
            for (j, a) in nzs:
                dt += a * s_vals[j]
            if dt <= 0.0:
                return 0.0 + 0.0j
        return complex(integrand_callable(*(list(s_vals) + free_ext_list)))

    return filtered


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

    re_val, _ = quad(f_re, lower, upper, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, lower, upper, **QUAD_OPTS)
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

    Returns (L, U). If the intersection is infeasible we return a
    DEGENERATE empty interval `(0.0, 0.0)` rather than a flipped
    (inf, -inf) sentinel. This matters because `scipy.quad(f, 0, 0)`
    returns 0 correctly, while `scipy.quad(f, +inf, -inf)` returns
    `-quad(f, -inf, +inf)` — the full real-line integral with a
    sign flip, which silently poisons any outer quadrature
    (`scipy.nquad`) that feeds this bounds function into
    `scipy.quad`. The 1D code path catches infeasible ranges
    up-front via `if L >= U: return 0`, so this degenerate form is
    indistinguishable from the old sentinel for the 1D path; but
    the 2D path needs the degenerate-empty form so that the inner
    integral returns 0 where the projection is empty.
    """
    L, U = -math.inf, math.inf
    infeasible = False
    for (a_int, c_eff) in s_constraints:
        a = a_int[s_index]
        if abs(a) < 1e-15:
            # Degenerate constraint (no dependence on s_index).  Under
            # Θ(0) = 0, the inequality is strict: `c_eff > 0` required.
            # Boundary c_eff = 0 is infeasible.
            if c_eff <= 0:
                infeasible = True
                break
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if infeasible or L >= U:
        return 0.0, 0.0   # degenerate empty interval
    return L, U


def _integrate_1d_polytope(integrand_callable, s_constraints, free_ext_vals):
    """Single integration variable. The polytope is an interval.

    A single axis is always cleanly bounded by the polytope (no deferred
    constraints possible since there's no inner axis to defer to), so
    the bounds are exact.  We still apply the Heaviside filter for
    uniformity with the 2D and nD paths and to guard against numerical
    edge cases.
    """
    L, U = _resolve_1d_bounds(s_constraints, s_index=0)
    if L >= U:
        return 0.0 + 0.0j
    # Heaviside-filtered integrand (see _make_heaviside_filtered_integrand
    # docstring).  For 1D this is a no-op since bounds are exact, but
    # enforcing the filter is cheap and future-proofs against any stale
    # residual after δ-sifting.
    filt = _make_heaviside_filtered_integrand(
        integrand_callable, s_constraints, free_ext_vals, m=1,
    )

    def f_re(s_0):
        return filt(s_0).real

    def f_im(s_0):
        return filt(s_0).imag

    from scipy.integrate import quad
    re_val, _ = quad(f_re, L, U, **QUAD_OPTS)
    try:
        im_val, _ = quad(f_im, L, U, **QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


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

    # ── Pre-split 2D constraints by role w.r.t. s_0 (Fix D) ──────
    # For each constraint `a_0 s_0 + a_1 s_1 + c_eff > 0` we
    # classify:
    #   pure_0      : a_1 ≈ 0, a_0 ≠ 0 → precomputed (L, U) on s_0.
    #   mixed       : both a_0 ≠ 0 and a_1 ≠ 0 → resolve per s_1 call.
    #   s1_only     : a_0 ≈ 0, a_1 ≠ 0 → pure residual inequality in s_1.
    #   constant    : a_0 ≈ 0, a_1 ≈ 0 → constant check (handled at build).
    pure_0_L = -math.inf
    pure_0_U = math.inf
    mixed_s0 = []       # list of (a_0, c_eff, a_1)
    s1_residual = []    # list of (a_1, c_eff): constraint reduces to a_1 s_1 + c_eff > 0
    bounds_s0_always_empty = False
    for (a_int, c_eff) in s_constraints:
        a_0 = float(a_int[0])
        a_1 = float(a_int[1])
        c_f = float(c_eff)
        if abs(a_0) < 1e-15 and abs(a_1) < 1e-15:
            # Constant constraint; Θ(0) = 0 wants c_eff > 0.
            if c_f <= 0.0:
                bounds_s0_always_empty = True
                break
            # Trivially satisfied — drop.
            continue
        if abs(a_0) < 1e-15:
            s1_residual.append((a_1, c_f))
            continue
        if abs(a_1) < 1e-15:
            bound = -c_f / a_0
            if a_0 > 0.0:
                if bound > pure_0_L:
                    pure_0_L = bound
            else:
                if bound < pure_0_U:
                    pure_0_U = bound
        else:
            mixed_s0.append((a_0, c_f, a_1))

    mixed_s0_t = tuple(mixed_s0)
    s1_residual_t = tuple(s1_residual)

    if bounds_s0_always_empty or pure_0_L >= pure_0_U:
        # Entire polytope empty.
        def bounds_s0(s_1_val):
            return 0.0, 0.0
    else:
        def bounds_s0(s_1_val):
            # Check s_1-only residual constraints (pure a_1 s_1 + c > 0).
            for (a_1, c_f) in s1_residual_t:
                if a_1 * s_1_val + c_f <= 0.0:
                    return 0.0, 0.0
            L = pure_0_L
            U = pure_0_U
            for (a_0, c_f, a_1) in mixed_s0_t:
                bound = -(c_f + a_1 * s_1_val) / a_0
                if a_0 > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            return L, U

    # Bounds on s_1: use constraints where a_0 = 0 AND a_1 != 0
    # (genuinely pure-s_1); else fall back to the ±OUTER_CAP
    # Heaviside-filtered quadrature domain.
    #
    # Subtle: a constraint with BOTH a_int[0] ≈ 0 and a_int[1] ≈ 0 is
    # NOT a bound on s_1 at all — it's a pure-external inequality that
    # either kills the polytope (c_eff ≤ 0 under Θ(0) = 0) or is
    # trivially satisfied (c_eff > 0).  Such constraints must NOT set
    # the `pure_s1_found` flag, otherwise we skip the OUTER_CAP fallback
    # and scipy.nquad runs on an unbounded axis, oversampling the
    # infinite-domain transform grid and biasing the integral.  At k=4
    # with distinct fields, δ-sifting can pin an integration variable
    # to external times and leave residual pure-external constraints
    # that trigger this path — the overshoot source for ~12% of
    # theory-vs-sim at k=4.
    L1, U1 = math.inf, -math.inf
    pure_s1_found = False
    tmp_L, tmp_U = -math.inf, math.inf
    for (a_int, c_eff) in s_constraints:
        if abs(a_int[0]) < 1e-15:
            a = a_int[1]
            if abs(a) < 1e-15:
                # Pure-external constraint: NOT a bound on s_1.
                # Θ(0) = 0: strict c_eff > 0 required; boundary infeasible.
                if c_eff <= 0:
                    return 0.0 + 0.0j
                continue
            # Genuine pure-s_1 constraint.
            pure_s1_found = True
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
        # Fallback cap: ±200 is ample with Heaviside-filtered integrand
        # (see _make_heaviside_filtered_integrand — correctness doesn't
        # depend on the cap being tight, only large enough to contain
        # the decaying tail).
        L1, U1 = -200.0, 200.0

    # Heaviside-filtered integrand: correctness no longer depends on
    # the (L1, U1) cap being exactly the true polytope projection.
    filt = _make_heaviside_filtered_integrand(
        integrand_callable, s_constraints, free_ext_vals, m=2,
    )

    def f_re(s_0, s_1):
        return filt(s_0, s_1).real

    def f_im(s_0, s_1):
        return filt(s_0, s_1).imag

    re_val, _ = nquad(f_re, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _integrate_nd_polytope(integrand_callable, s_constraints, free_ext_vals, m):
    """
    General `m >= 3` case via `scipy.integrate.nquad` with nested
    bound functions.

    Variable ordering for nquad: the FIRST argument of the integrand
    is the INNERMOST integration variable.  We integrate s_0 first
    (innermost), then s_1, ..., up to s_{m-1} (outermost).

    For each variable s_k, its bounds are computed by:
      1. Substituting all OUTER variables (s_{k+1}, ..., s_{m-1}) and
         the external times into the linear constraints
         `a · s + c_eff > 0`.
      2. Resolving the remaining 1D polytope on s_k via
         `_resolve_1d_bounds`.
    """
    from scipy.integrate import nquad

    def _make_bound_fn(k_var):
        """Return a function `bounds_k(*outer_vals)` that returns (L, U)
        for variable s_{k_var} given all the OUTER variable values.

        nquad calls bound functions with the outer variables in
        REVERSE order (s_{k+1}, s_{k+2}, ..., s_{m-1}).

        Important subtlety — constraints that still couple to a MORE-
        INNER axis s_j (j < k_var) must be SKIPPED here.  Those
        constraints are genuine bounds on the inner variable, not on
        s_{k_var}, and they will be resolved when nquad nests into the
        deeper (smaller-index) integration axis and s_j becomes the
        resolution target.  Failing to filter them is a latent bug:
        `_resolve_1d_bounds` inspects only `a_int[s_index]` and treats
        `|a_int[k_var]| < 1e-15` as a pure-residual check, which then
        spuriously declares the polytope infeasible whenever the
        accumulated residual (still containing the unresolved inner
        coefficient) is negative.  This mirrors the filter
        `abs(a_int[0]) < 1e-15` that `_integrate_2d_polytope` already
        applies to the outer-axis bound (lines ~1240–1255).

        Regression: test_phase_J_nd_polytope_preserves_deferred_constraints
        in tests/test_time_domain.py.

        Fix D (2026-04-21): constraint classification and sparse-
        coefficient extraction happen ONCE at closure-build time, not
        on every call.  Constraints with only `s_{k_var}` nonzero
        (no outer coupling) contribute a fixed slice to (L, U) that we
        precompute.  The per-call path now only iterates constraints
        whose bound actually varies with the outer values.
        """
        # ── Classify constraints by their role w.r.t. axis k_var ──
        # Four buckets (after filtering deferred-inner constraints):
        #   pure_k      : a[k_var] != 0, no outer coupling → precomputed
        #                 contribution to (L, U).
        #   mixed       : a[k_var] != 0, some outer axes coupled →
        #                 resolve per call using outer vals.
        #   outer_only  : a[k_var] == 0, some outer axes coupled →
        #                 pure residual inequality; kills polytope when
        #                 negative at this outer point.
        #   (trivial satisfied or trivially infeasible constraints
        #    are resolved at build time.)
        pure_k_L = -math.inf
        pure_k_U = math.inf
        mixed = []         # list of (a_k, c_eff, outer_pairs)
        outer_only = []    # list of (c_eff, outer_pairs)
        infeasible_at_build = False
        for (a_int, c_eff) in s_constraints:
            # Skip constraints that still couple to a more-inner axis.
            deferred = False
            for j in range(k_var):
                if abs(a_int[j]) >= 1e-15:
                    deferred = True
                    break
            if deferred:
                continue
            a_k = float(a_int[k_var])
            # Sparse outer-axis coefficients: tuple of (outer_index, coeff)
            # where outer_index is the position in *outer_vals (not the
            # absolute axis index).  nquad passes outer_vals with
            # outer_vals[0] = s_{k_var+1}, [1] = s_{k_var+2}, etc.,
            # which matches the original `j = k_var + 1 + i_outer`
            # indexing.
            outer_pairs = tuple(
                (j - (k_var + 1), float(a_int[j]))
                for j in range(k_var + 1, len(a_int))
                if abs(float(a_int[j])) >= 1e-15
            )
            c_eff_f = float(c_eff)
            if abs(a_k) < 1e-15:
                if not outer_pairs:
                    # Constant constraint: Θ(0)=0 wants c_eff > 0.
                    if c_eff_f <= 0.0:
                        infeasible_at_build = True
                        break
                    # Trivially satisfied → drop.
                    continue
                outer_only.append((c_eff_f, outer_pairs))
            else:
                if outer_pairs:
                    mixed.append((a_k, c_eff_f, outer_pairs))
                else:
                    bound = -c_eff_f / a_k
                    if a_k > 0.0:
                        if bound > pure_k_L:
                            pure_k_L = bound
                    else:
                        if bound < pure_k_U:
                            pure_k_U = bound

        # If the polytope is empty under Θ(0)=0 irrespective of outer
        # values, or the pure-k_var bounds are crossed, return a
        # constant-zero bounds function — scipy.quad(a, a) is 0, so
        # this cleanly kills the outer integral over the empty region.
        if infeasible_at_build or pure_k_L >= pure_k_U:
            def _bounds_infeas(*outer_vals):
                return 0.0, 0.0
            return _bounds_infeas

        mixed_t = tuple(mixed)
        outer_only_t = tuple(outer_only)

        def bounds_k(*outer_vals):
            # Start from the precomputed pure-axis bounds (may be ±inf
            # if no pure-k constraint is present on that side — mixed
            # constraints below may finitely bound the axis instead).
            L = pure_k_L
            U = pure_k_U
            # Check outer-only constraints first (cheap polytope kill).
            for c_eff, outer_pairs in outer_only_t:
                val = c_eff
                for (oi, a) in outer_pairs:
                    val += a * outer_vals[oi]
                if val <= 0.0:
                    return 0.0, 0.0
            # Apply mixed (outer-coupled on-axis) constraints.
            for a_k, c_eff, outer_pairs in mixed_t:
                total_c = c_eff
                for (oi, a) in outer_pairs:
                    total_c += a * outer_vals[oi]
                bound = -total_c / a_k
                if a_k > 0.0:
                    if bound > L:
                        L = bound
                else:
                    if bound < U:
                        U = bound
            if L >= U:
                return 0.0, 0.0
            # Cap infinite bounds only when still unbounded.  Matches
            # the original post-`_resolve_1d_bounds` behaviour where a
            # cap never clips a finite constraint-derived bound.
            if math.isinf(L):
                L = -OUTER_CAP
            if math.isinf(U):
                U = OUTER_CAP
            return L, U
        return bounds_k

    # Fallback cap for any axis whose bounds resolve to (-inf, +inf).
    # With the Heaviside-filtered integrand (below), correctness does
    # NOT depend on this cap being tight — the filter kills any
    # contribution from the region outside the true polytope.  The cap
    # only needs to be large enough that the decaying integrand has
    # effectively vanished by the boundary.  For retarded propagators
    # with time constant τ, ~10τ gives exp(-10) ≈ 5e-5 tail.  Hawkes
    # τ=10 ⟹ ±200 is ample.  A wider cap is harmless (the filter
    # zeros out the extra volume) but slightly slower to quadrature
    # through.
    OUTER_CAP = 200.0

    # Outermost variable s_{m-1}: bounds computed from constraints with
    # zero coefficient on all inner variables (pure-s_{m-1}).  If no
    # such constraints exist, fall back to ±OUTER_CAP.
    L_out, U_out = _outer_bounds(s_constraints, m - 1)
    if L_out >= U_out:
        return 0.0 + 0.0j
    if not math.isfinite(L_out):
        L_out = -OUTER_CAP
    if not math.isfinite(U_out):
        U_out = OUTER_CAP

    # Build the list of bound specifications for nquad.
    # Order: innermost first.  For variables s_0, ..., s_{m-2}, use
    # callable bounds (functions of outer vars).  For s_{m-1}, use the
    # constant tuple (L_out, U_out).
    bound_specs = [_make_bound_fn(k) for k in range(m - 1)]
    bound_specs.append((L_out, U_out))

    # Heaviside-filtered integrand: evaluates to 0 outside the true
    # polytope, guarding against any cap overshoot or deferred-
    # constraint leak.
    filt = _make_heaviside_filtered_integrand(
        integrand_callable, s_constraints, free_ext_vals, m=m,
    )

    def f_re(*all_args):
        # nquad passes integration variables (s_0, ..., s_{m-1}).
        return filt(*all_args).real

    def f_im(*all_args):
        return filt(*all_args).imag

    re_val, _ = nquad(f_re, bound_specs, opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, bound_specs, opts=QUAD_OPTS)
    except Exception:
        im_val = 0.0
    return complex(re_val, im_val)


def _outer_bounds(s_constraints, k_var):
    """Compute bounds on s_{k_var} from constraints whose coefficients on
    all OTHER integration variables vanish (pure-s_{k_var} constraints).

    Used for the outermost integration variable in nD polytope
    integration, where there are no further outer variables to
    substitute.  If no pure constraint exists, returns (-inf, +inf).
    """
    L, U = -math.inf, math.inf
    pure_found = False
    for (a_int, c_eff) in s_constraints:
        # Check that a_int is purely k_var-dependent
        is_pure = True
        for j in range(len(a_int)):
            if j == k_var:
                continue
            if abs(a_int[j]) >= 1e-15:
                is_pure = False
                break
        if not is_pure:
            continue
        pure_found = True
        a = a_int[k_var]
        if abs(a) < 1e-15:
            # Degenerate pure-k_var constraint.  Θ(0) = 0: boundary
            # c_eff = 0 is infeasible; strict c_eff > 0 required.
            if c_eff <= 0:
                return math.inf, -math.inf  # infeasible
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if pure_found and L >= U:
        return math.inf, -math.inf
    if not pure_found:
        # No pure constraints — let inner bounds clip via callables.
        return -math.inf, math.inf
    return L, U


# ───────────────────────────────────────────────────────────────────────
# Edge → propagator-index lookup (matches _resolve_edge_propagator_data)
# ───────────────────────────────────────────────────────────────────────

def format_td_integral_latex(
    tree_result,
    typed_diagram=None,
    combined_prefactor=None,
    ext_time_vars=None,
    label=None,
):
    r"""
    Build a LaTeX string describing the time-domain integral that
    Phase J evaluates for a given tree diagram, in the same spirit as
    the notebook's `show_integral` helper for the frequency-domain
    Phase I integrand.

    This helper is intended for debugging and documentation only —
    numerical evaluation goes through `tree_result['contribution']`.

    Parameters
    ----------
    tree_result : dict
        Output of `integrate_tree_diagram`. Must contain the
        `edge_info`, `integration_vars`, and `constraints` keys.
    typed_diagram : TypedDiagram, optional
        Used for the leaf/internal-vertex display. If not supplied the
        vertex-time assignment is inferred from `tree_result`.
    combined_prefactor : SR or numeric, optional
        Prefactor to display in front of the integral. If None, the
        prefactor is folded into the integrand and not displayed
        separately.
    ext_time_vars : list of SR, optional
        External time symbols, used for display only. If not provided,
        default labels `t_1, t_2, ...` are inferred.
    label : str, optional
        A short label for the diagram (e.g., 'Tree-1', 'Hawkes-k2').
        Rendered as a prefix in the output.

    Returns
    -------
    str
        A LaTeX string, ready to pass to `display(Math(...))` in a
        Jupyter notebook. Includes:
          - vertex-time assignment block
          - the integral expression
            `C_Γ = pref · ∫ ds_1 ... ds_m  ∏ G_R[…](…)`
          - the retarded polytope constraints `Θ(t_v − t_u)`
          - any δ-edge summary (edges with a nonzero δ coefficient)
    """
    from sage.all import latex

    edge_info = tree_result.get('edge_info', [])
    integration_vars = tree_result.get('integration_vars', [])
    constraints = tree_result.get('constraints', [])

    lines = []
    if label:
        lines.append(r'\text{' + str(label) + r'} \;:')

    # Integration variables
    if integration_vars:
        int_vars_tex = r' \wedge '.join(latex(v) for v in integration_vars)
        lines.append(
            r'\text{non-leaf vertex times: } \{' + int_vars_tex + r'\}'
        )
        integrals_tex = ''
        for v in integration_vars:
            integrals_tex += r'\int\!d' + latex(v) + r'\;'
    else:
        integrals_tex = ''

    # Combined prefactor.  For diagrams with a non-local cumulant
    # kernel (NoiseSourceType vertex), the substituted, simplified
    # τ-dependent prefactor lives in ``tree_result['cumulant_prefactor']``
    # and goes INSIDE the integral as a kernel factor.  For ordinary
    # diagrams the kappa machinery is absent and the user-supplied
    # ``combined_prefactor`` (τ-independent scalar) goes OUTSIDE.
    pref_tex = ''
    inside_kernel_tex = ''
    cumulant_pref = tree_result.get('cumulant_prefactor', None)
    has_cumulant = tree_result.get('has_cumulant_kernel', False)
    if has_cumulant and cumulant_pref is not None:
        # The cumulant prefactor is τ-dependent — render it inside the
        # integral, parenthesised, before the propagator product.
        inside_kernel_tex = (
            r'\bigl[' + latex(SR(cumulant_pref)) + r'\bigr] \cdot '
        )
    elif combined_prefactor is not None:
        pref_tex = latex(combined_prefactor) + r'\;'

    # Build the edge-factor product
    factor_bits = []
    delta_summary_bits = []
    for ei in edge_info:
        u = ei['u']
        v = ei['v']
        ri = ei['ri']
        pi = ei['pi']
        dt = ei['dt_sym']
        delta_c = ei.get('delta_coeff', 0)
        # G_R[pi, ri](dt)
        factor_bits.append(
            r'G^{R}_{' + str(pi) + ',' + str(ri) + r'}\!\bigl('
            + latex(dt) + r'\bigr)'
        )
        try:
            if abs(complex(delta_c)) > 1e-15:
                delta_summary_bits.append(
                    r'c_{' + str(pi) + ',' + str(ri) + r'} = '
                    + latex(delta_c)
                )
        except Exception:
            pass

    product_tex = r' \cdot '.join(factor_bits) if factor_bits else r'1'

    # Main equation: C_Γ = pref × ∫ds × [κ(τ)] × ∏ G_R × Θ
    lines.append(
        r'C_{\Gamma}(t) \;=\; '
        + pref_tex
        + integrals_tex
        + inside_kernel_tex
        + product_tex
    )

    # Retardation constraints (as Θ factors)
    if constraints:
        theta_bits = [
            r'\Theta\bigl(' + latex(SR(c)) + r'\bigr)'
            for c in constraints
        ]
        lines.append(
            r'\text{retardation: } '
            + r' \cdot '.join(theta_bits)
        )

    # δ-edge summary, if any
    if delta_summary_bits:
        lines.append(
            r'\text{instantaneous } \delta\text{-edge coefficients: } '
            + r', \; '.join(delta_summary_bits)
        )

    n_subsets = tree_result.get('n_subsets_evaluated')
    n_skipped = tree_result.get('n_shotnoise_skipped', 0)
    if n_subsets is not None:
        lines.append(
            r'\text{δ-edge subsets evaluated: } '
            + str(n_subsets)
            + (
                r'\;\;(\text{shot-noise skipped: } '
                + str(n_skipped) + r')'
                if n_skipped else ''
            )
        )

    return r' \\ '.join(lines)


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


# ───────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ───────────────────────────────────────────────────────────────────────
# ``integrate_tree_diagram`` was the original name for what is now
# ``integrate_diagram`` (generalised to any loop order).  Keep the old
# name resolvable so tests and external callers don't break.
integrate_tree_diagram = integrate_diagram

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
    'limit': 50,       # max subintervals (scipy default: 50; old value: 200)
    'epsrel': 1e-4,    # relative tolerance (scipy default: 1.49e-8)
    # 'epsabs': ...    # absolute tolerance (leave unset to use scipy default)
}


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

    # ── 1. Numerical G(t) matrix (smooth + delta parts) ──────────
    t_sym = SR.var('_t_td_')
    G_t_obj = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

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

    # ── 3. Gather per-edge info: ri, pi, dt, delta_coeff, smooth factor
    edges = list(D.edges())
    edge_info = []
    for (u, v, lbl) in edges:
        ri, pi = _lookup_prop_indices(typed_diagram, (u, v, lbl))
        dt = SR(vertex_time[v] - vertex_time[u])
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
            if num_params:
                subset_factor_delta = subset_factor_delta.subs(num_params)
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
        if num_params:
            subset_factor = subset_factor.subs(num_params)
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
                integrand_fc_sub, subset_constraint_data, m_sub,
            )
        )
        subset_diagnostics.append({
            'delta_edges': delta_edges,
            'smooth_edges': smooth_edges,
            'status': 'evaluated',
            'm_after_delta': m_sub,
        })

    # ── Build the final contribution callable ─────────────────────
    def contribution(*ext_time_values):
        if len(ext_time_values) != len(ext_time_vars):
            raise ValueError(
                f"contribution() expects {len(ext_time_vars)} positional "
                f"arguments (one per ext_time_var); got "
                f"{len(ext_time_values)}."
            )
        free_vals = [float(ext_time_values[j]) for j in free_ext_idx]
        total = 0.0 + 0.0j
        for cfn in subset_contributions:
            total = total + complex(cfn(free_vals))
        return total

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

    re_val, _ = nquad(f_re, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
    try:
        im_val, _ = nquad(f_im, [bounds_s0, (L1, U1)], opts=QUAD_OPTS)
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

    # Combined prefactor (optional)
    pref_tex = ''
    if combined_prefactor is not None:
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

    # Main equation: C_Γ = pref × ∫ds × ∏ G_R × Θ
    lines.append(
        r'C_{\Gamma}(t) \;=\; '
        + pref_tex
        + integrals_tex
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

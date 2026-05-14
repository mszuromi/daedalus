"""
msrjd.integration.time_domain.grouped_integral
==============================================
Prediagram-grouped vertex-time integration: a prototype of the
"sum integrands first, integrate once" optimisation.

Why
---
Every typed diagram emitted by the enumerator carries its parent
``prediagram`` tag.  Typed diagrams sharing the same prediagram have
**identical graph topology, leaves, internal vertices, and integration
variables**.  They differ only in per-edge propagator field-type
indices ``(ri, pi)`` and per-vertex coefficients.

The current ``integrate_diagram`` evaluates each typed diagram in
isolation, repaying the full per-diagram setup cost (graph walk,
vertex-time symbol allocation, 2^|E| subset bookkeeping, fast_callable
compile, …) once per typed diagram.  For ``(k=2, ell=1)`` multipop the
user saw 7736 typed diagrams across 9 prediagrams — ~860 diagrams per
prediagram, all paying full setup cost.

This module groups by prediagram and:

1. Builds the prediagram-level scaffolding ONCE (graph, vertex-time
   map, integration vars, Wick contraction mapping, compensation
   factor, the 2^|E| subset structure).
2. For each subset, builds the COMBINED stripped integrand
   ``Σ_td cp_td · Π_e factor_e^{(td)}`` and JIT-compiles a single
   ``fast_callable`` from it.
3. Reuses the per-subset polytope and pole/residue extraction once.

Math
----
By linearity of integration,
::

    Σ_td ∫ ds ⋯ td-integrand_td   ==   ∫ ds ⋯ Σ_td td-integrand_td

The integration domain is identical across all typed diagrams of a
single prediagram, so combining is exact (not an approximation).

Scope
-----
Drop-in for ``integrate_diagram`` at the "summed contribution" level —
returns the SAME shape as the per-diagram path: one ``contribution``
callable ``f(*ext_time_values) -> complex`` per group, which is the
sum across all typed diagrams in the group.

Limitations of this MVP
-----------------------
- Wick contraction enumeration assumes all typed diagrams in the group
  share the same ``external_legs`` mapping (each prediagram leaf has
  the same external-field type across the group).  Holds for every
  k=2 case I've seen because the external fields are fixed by the
  user; if a future enumerator emits different external-leg
  assignments for the same prediagram, regroup at a finer key
  (prediagram × external_legs).
- Subsets with m ∈ {0, 1} still use scipy.nquad on the summed
  ``fast_callable``.  The analytic merged-residue path (below) is
  wired for m=2 (polygon, Stage 3a) and m≥3 (causal-poset chain
  simplex, Stage 3b); m=0/1 would need a separate pole-residue
  closure plus an interval-of-integration analytic form for m=1.
- Shot-noise δ-spikes (residual external-time equalities after δ-
  elimination) are skipped — they contribute zero to ``total_C`` and
  the per-diagram path's ``delta_contributions`` accounting isn't
  re-built across the group.

Analytic merged-residue path (Stage 4a, opt-in via
``USE_GROUPED_ANALYTIC_MODESUM``)
---------------------------------------------------------------------
For each per-subset polytope with m ∈ {2, ≥3}, the grouped path
builds a merged-residue tensor

    B_α = Σ_td (cp_td · Π_δedge δ_coeff_td) · Π_smooth_edge C^{(td)}_{α_e, e}

over multi-indices α = (α_1, …, α_E) of per-edge poles, then routes
through the existing per-diagram analytic evaluators
(``_integrate_2d_polygon_modesum`` for m=2,
``_integrate_nd_polytope_poset_modesum`` for m≥3) via a
``pole_tuples`` iterator override.  Each evaluator only ever needed
``(coefficient, lambdas)`` per multi-index; for the per-diagram path
the coefficient factorises as ``Π_e C_α_e`` (Cartesian product),
whereas the grouped path replaces it with the summed tensor B_α —
the underlying analytic integration is identical.  The precision
floor drops from scipy.nquad's ~1e-8 (default ``epsrel``) to machine
ε (~1e-15).

The pole spectrum (``propagator_data['pole_vals']``) is identical
across all td's in the group; only the residues differ via the per-td
``(ri, pi)`` edge indices.  When non-local cumulant kernels are
present (any ``vertex_leg_time`` is non-empty) the integrand contains
non-rational factors and the analytic path cannot fire — the
fast_callable + scipy fallback runs unchanged.

If anything other than the simple Hawkes / multipop tree+1-loop case
is exercised, this prototype should be carefully cross-checked against
``integrate_diagram`` (sum N per-diagram results vs 1 grouped result —
they MUST match to floating-point round-off).
"""

from sage.all import SR, fast_callable, CDF, solve as sage_solve

from msrjd.integration.time_domain.propagator_td import (
    build_G_t_matrix,
    G_t_entry,
    G_t_delta_coeff,
)
from msrjd.integration.time_domain.final_integral import (
    _lookup_prop_indices,
    _integrate_polytope,
    _integrate_2d_polygon_modesum,
    _integrate_nd_polytope_poset_modesum,
    _loop_number_from_graph,
    EdgeModeSum,
    TAU_KERNEL_CAP,
)
# Module reference so per-call reads pick up notebook overrides of
# ``USE_POLYGON_M2_INTEGRATOR`` / ``USE_POSET_INTEGRATOR``.
from msrjd.integration.time_domain import final_integral as _fi_mod
from msrjd.core.vertices import NoiseSourceType


# Master switch for the analytic merged-residue path inside
# ``integrate_grouped_diagram``.  When True (and the underlying
# polygon/poset stage flags are also on), each per-subset analytic
# call gets a ``pole_tuples`` iterator with merged residues
# (B_α = Σ_td cp_td · Π_e C^{(td)}_{α_e, e}), giving machine
# precision; when False, every subset falls through to scipy.nquad
# (~1e-8 rel).
USE_GROUPED_ANALYTIC_MODESUM = True

# Overflow guard threshold (matches ``EXP_REAL_LIMIT`` in
# final_integral.py).  ``cmath.exp(z.real > 709)`` raises
# OverflowError on IEEE doubles; clip at 600 for headroom.
_GROUPED_EXP_REAL_LIMIT = 600.0


def _evaluate_grouped_m0_modesum(
    pole_tuples,
    subset_constraint_data,
    free_ext_vals,
):
    """Analytic merged-residue evaluation for m=0 (no integration vars).

    Returns ``Σ_α B_α · exp(Σ_e λ_α_e · Δt_e)`` after checking each
    smooth-edge constraint ``Δt_e > 0`` (Θ(0) = 0 convention).  Returns
    ``0`` if any constraint is violated, ``None`` on overflow.
    """
    import cmath
    for (a_int, a_ext, c0) in subset_constraint_data:
        c_eff = float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        if c_eff <= 0:
            return 0.0 + 0.0j
    # ``a_int`` is empty for m=0; Δt_e = c_eff.
    dt_per_edge = [
        float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        for (_a_int, a_ext, c0) in subset_constraint_data
    ]
    total = 0.0 + 0.0j
    for (B_alpha, lambdas) in pole_tuples:
        gamma = 0.0 + 0.0j
        for e in range(len(lambdas)):
            gamma += lambdas[e] * dt_per_edge[e]
        if abs(gamma.real) > _GROUPED_EXP_REAL_LIMIT:
            return None
        try:
            total += B_alpha * cmath.exp(gamma)
        except (OverflowError, ValueError):
            return None
    return total


def _integrate_grouped_m1_modesum(
    pole_tuples,
    subset_constraint_data,
    free_ext_vals,
    bbox_cap,
):
    r"""Analytic merged-residue 1D integral.

    Computes ``∫_L^U Σ_α B_α · exp(α_s · s + γ_α) ds`` where
        α_s = Σ_e λ_α_e · a_int_e[0]
        γ_α = Σ_e λ_α_e · c_ext_e,  c_ext_e = c0_e + Σ_j a_ext_e[j]·t_free[j]
    and ``[L, U]`` is the polytope interval determined by the smooth-
    edge constraints ``a_int_e[0]·s + c_ext_e > 0``.  When no
    constraint bounds the integration variable on one side, the
    corresponding L or U is treated as ±∞ — the boundary term in the
    closed form is set to zero whenever the integrand decays in that
    direction (``sign(Re α_s)`` matches), else the caller falls back
    to scipy.nquad (which handles the unbounded endpoint with the
    standard adaptive-quadrature substitution).  Clipping at
    ±bbox_cap would otherwise leave a residual ``exp(α_s · bbox_cap)``
    contribution at the ~1e-9 level for typical Hawkes timescales —
    measured on ``quad_exp_k2_ell0`` before this fix.

    Returns ``0`` if the interval is empty / infeasible, ``None`` on
    overflow or unbounded interval with non-decaying integrand.
    """
    import cmath
    import math
    # Resolve the feasible interval — track unbounded sides exactly.
    L = -math.inf
    U = +math.inf
    for (a_int, a_ext, c0) in subset_constraint_data:
        a = float(a_int[0]) if a_int else 0.0
        c_eff = float(c0) + sum(
            float(a_ext[i]) * float(free_ext_vals[i])
            for i in range(len(a_ext))
        )
        if abs(a) < 1e-15:
            if c_eff <= 0:
                return 0.0 + 0.0j
            continue
        bound = -c_eff / a
        if a > 0:
            if bound > L:
                L = bound
        else:
            if bound < U:
                U = bound
    if L >= U:
        return 0.0 + 0.0j
    L_inf = math.isinf(L)
    U_inf = math.isinf(U)

    # Sum the analytic 1D exponential integral term-by-term.
    total = 0.0 + 0.0j
    for (B_alpha, lambdas) in pole_tuples:
        alpha_s = 0.0 + 0.0j
        gamma = 0.0 + 0.0j
        for e, (a_int, a_ext, c0) in enumerate(subset_constraint_data):
            lam = lambdas[e]
            a_int_v = float(a_int[0]) if a_int else 0.0
            c_ext = float(c0) + sum(
                float(a_ext[i]) * float(free_ext_vals[i])
                for i in range(len(a_ext))
            )
            alpha_s += lam * a_int_v
            gamma += lam * c_ext
        if abs(gamma.real) > _GROUPED_EXP_REAL_LIMIT:
            return None
        try:
            if abs(alpha_s) < 1e-15:
                # α_s ≈ 0: integrand is constant in s.
                if L_inf or U_inf:
                    # Integral diverges; bail to scipy.nquad fallback.
                    return None
                contrib = (U - L) * cmath.exp(gamma)
            else:
                # ∫ exp(α_s · s + γ) ds = (exp(α_s·U+γ) - exp(α_s·L+γ)) / α_s
                # At ±∞ the term vanishes iff sign(Re α_s) matches.
                if U_inf:
                    if alpha_s.real >= 0:
                        return None  # would diverge at +∞
                    term_U = 0.0 + 0.0j
                else:
                    arg = alpha_s * U + gamma
                    if abs(arg.real) > _GROUPED_EXP_REAL_LIMIT:
                        return None
                    term_U = cmath.exp(arg)
                if L_inf:
                    if alpha_s.real <= 0:
                        return None  # would diverge at -∞
                    term_L = 0.0 + 0.0j
                else:
                    arg = alpha_s * L + gamma
                    if abs(arg.real) > _GROUPED_EXP_REAL_LIMIT:
                        return None
                    term_L = cmath.exp(arg)
                contrib = (term_U - term_L) / alpha_s
        except (OverflowError, ValueError):
            return None
        total += B_alpha * contrib
    return total


def integrate_grouped_diagram(
    typed_diagrams,
    combined_prefactors,
    propagator_data,
    ext_time_vars,
    num_params=None,
    origin_leaf_idx=0,
    external_fields=None,
):
    """
    Sum-then-integrate over a group of typed diagrams sharing one prediagram.

    Parameters
    ----------
    typed_diagrams : list[TypedDiagram]
        Length-N list.  **Must all share the same ``prediagram``**.  The
        function asserts this and bails on mismatch.
    combined_prefactors : list[SR or numeric]
        Length-N list.  Per-typed-diagram full prefactor (symmetry
        factor + multiplicity, as returned by
        ``classify_coefficient_factors``).
    propagator_data : dict
        Same shape as the per-diagram entry point.  Must contain
        ``pole_vals``, ``C_mats``, optionally ``D_delta``.
    ext_time_vars, num_params, origin_leaf_idx, external_fields :
        Same as ``integrate_diagram``.

    Returns
    -------
    dict with keys
        ``'status'`` — ``'ok'`` / ``'empty_polytope'`` / ``'failed'``.
        ``'contribution'`` — ``f(*ext_time_values) -> complex``, the
          summed contribution across the entire group.
        ``'n_diagrams'`` — N (input length, for bookkeeping).
        ``'integration_vars'`` — symbolic list (for debugging).
        ``'reason'`` — failure message if applicable.

    Notes
    -----
    The returned ``contribution`` is the SAME shape as
    ``integrate_diagram``'s, so a wrapper can swap between per-diagram
    and grouped paths without touching downstream code.
    """
    import itertools as _itertools
    from math import factorial as _factorial

    if not typed_diagrams:
        return {
            'status': 'failed', 'contribution': None,
            'n_diagrams': 0,
            'reason': 'empty typed_diagrams list',
        }
    if len(typed_diagrams) != len(combined_prefactors):
        return {
            'status': 'failed', 'contribution': None,
            'n_diagrams': len(typed_diagrams),
            'reason': (
                f'len(typed_diagrams)={len(typed_diagrams)} != '
                f'len(combined_prefactors)={len(combined_prefactors)}'
            ),
        }

    # All typed diagrams in the group must share the same prediagram
    # graph object.  We use ``id(td.prediagram[0])`` as the identity
    # check — Python identity is fine because the enumerator emits
    # the same Graph object reference for typed diagrams of one
    # prediagram.
    pd0 = typed_diagrams[0].prediagram
    if not all(td.prediagram is pd0 for td in typed_diagrams):
        return {
            'status': 'failed', 'contribution': None,
            'n_diagrams': len(typed_diagrams),
            'reason': (
                'typed_diagrams in the group must all share the same '
                'prediagram (by identity); got mixed prediagrams.'
            ),
        }

    # ── Shared scaffolding (prediagram-level) ─────────────────────
    td0 = typed_diagrams[0]
    loop_number = _loop_number_from_graph(td0)

    D = td0.prediagram[0]
    leaves = list(td0.prediagram[2])
    leaf_set = set(leaves)

    if len(ext_time_vars) != len(leaves):
        return {
            'status': 'failed', 'contribution': None,
            'n_diagrams': len(typed_diagrams),
            'reason': (
                f"ext_time_vars has length {len(ext_time_vars)} but "
                f"the diagram has {len(leaves)} leaves."
            ),
        }

    t_sym = SR.var('_t_td_')
    G_t_obj = build_G_t_matrix(propagator_data, t_sym, num_params=num_params)

    # ── Wick contraction enumeration (shared across the group) ────
    # ``external_legs`` is per-typed-diagram, but in practice every td
    # in the group has the SAME external_legs mapping (user fixes the
    # external fields; the enumerator emits the same leaf-to-field
    # assignment for typed diagrams of the same prediagram, modulo the
    # leaf permutations the Wick contraction itself enumerates here).
    # We use td0's external_legs as the canonical reference.
    if external_fields is not None and len(external_fields) == len(leaves):
        _leaf_fields = [td0.external_legs.get(lf) for lf in leaves]
        _cp_by_field = {}
        _leaves_by_field = {}
        for cp, field in enumerate(external_fields):
            _cp_by_field.setdefault(field, []).append(cp)
        for j, field in enumerate(_leaf_fields):
            _leaves_by_field.setdefault(field, []).append(j)

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

        # Within-vertex compensation factor (per td0; assumed shared
        # across the group since external_legs is shared).
        _vertex_of_leaf = {}
        for ek in td0.edge_types:
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
            field = td0.external_legs.get(lf)
            _vertex_field_counts.setdefault(v, {}).setdefault(field, 0)
            _vertex_field_counts[v][field] += 1
        _compensation = 1
        for v, fcounts in _vertex_field_counts.items():
            for field, count in fcounts.items():
                _compensation *= _factorial(count)
    else:
        _all_mappings = [{j: j for j in range(len(leaves))}]
        _compensation = 1

    # Build vertex_time for the first mapping (shared).
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

    # ── Per-typed-diagram noise-source per-leg time maps ──────────
    # Each td may have its own NoiseSourceType vertex assignments, so
    # the per-leg time map is computed PER td.  In practice for the
    # heterogeneous Hawkes cases this map is empty (no non-local
    # cumulant kernels), so this loop is a no-op there.
    per_td_vertex_leg_time = []
    per_td_noise_source_specs = []
    per_td_extra_tau_syms = []
    for td_idx, td in enumerate(typed_diagrams):
        vertex_leg_time = {}
        noise_source_specs = {}
        extra_tau_syms = []
        vertex_assignments = (
            getattr(td, 'vertex_assignments', None) or {}
        )
        for v, vtype in vertex_assignments.items():
            if not isinstance(vtype, NoiseSourceType):
                continue
            if not vtype.cumulant_specs:
                continue
            legs0 = vtype.cumulant_specs[0]['legs']
            anchor_leg = legs0[0]
            other_leg  = legs0[1] if len(legs0) > 1 else legs0[0]
            if anchor_leg == other_leg:
                other_leg = anchor_leg
            anchor_time = vertex_time[v]
            tau_sym = SR.var(
                f's_v{v}_tau_g{td_idx}_td_',
                latex_name=rf'\tau_{{v_{{{v}}}}}^{{({td_idx})}}',
            )
            extra_tau_syms.append((tau_sym, v))
            vertex_leg_time[v] = {
                anchor_leg: anchor_time,
                other_leg:  anchor_time - tau_sym,
            }
            noise_source_specs[v] = list(vtype.cumulant_specs)
        per_td_vertex_leg_time.append(vertex_leg_time)
        per_td_noise_source_specs.append(noise_source_specs)
        per_td_extra_tau_syms.append(extra_tau_syms)

    # Union of all extra τ symbols across the group — each contributes
    # its own integration variable.  Plain (no-noise-source) groups
    # see an empty list and the original integration_vars is used.
    grouped_extra_tau = []
    for lst in per_td_extra_tau_syms:
        for entry in lst:
            grouped_extra_tau.append(entry)
    integration_vars_grouped = list(integration_vars) + [
        ts for (ts, _v) in grouped_extra_tau
    ]

    # ── Per-typed-diagram edge_info + prefactor (the only per-td work)
    # For each td: ri, pi per edge → delta_coeff, smooth_factor; then
    # the combined prefactor with kernel substitution (if applicable).
    per_td_edge_info = []
    per_td_cp = []
    for td_idx, td in enumerate(typed_diagrams):
        vertex_leg_time = per_td_vertex_leg_time[td_idx]
        edges = list(D.edges())
        edge_info = []
        for (u, v, lbl) in edges:
            ri, pi = _lookup_prop_indices(td, (u, v, lbl))
            if u in vertex_leg_time:
                edge_resp_leg, _ = td.edge_types[(u, v, lbl)]
                edge_pop_idx = edge_resp_leg[1] - 1
                t_u = vertex_leg_time[u].get(
                    edge_pop_idx, vertex_time[u]
                )
            else:
                t_u = vertex_time[u]
            if v in vertex_leg_time:
                edge_resp_leg_v, _ = td.edge_types[(u, v, lbl)]
                edge_pop_idx_v = edge_resp_leg_v[1] - 1
                t_v = vertex_leg_time[v].get(
                    edge_pop_idx_v, vertex_time[v]
                )
            else:
                t_v = vertex_time[v]
            dt = SR(t_v - t_u)
            delta_c = G_t_delta_coeff(G_t_obj, pi, ri)
            smooth_factor = G_t_entry(G_t_obj, pi, ri, dt,
                                       include_heaviside=False)
            edge_info.append({
                'u': u, 'v': v, 'lbl': lbl,
                'ri': ri, 'pi': pi,
                'dt_sym': dt,
                'delta_coeff': delta_c,
                'smooth_factor': smooth_factor,
            })
        per_td_edge_info.append(edge_info)

        cp = (SR(combined_prefactors[td_idx])
              if combined_prefactors[td_idx] is not None else SR(1))
        if num_params:
            cp = cp.subs(num_params)

        # Non-local cumulant kernel substitution (per td, if needed).
        ns_specs = per_td_noise_source_specs[td_idx]
        ex_tau   = per_td_extra_tau_syms[td_idx]
        if ns_specs:
            kappa_subs = {}
            for v, specs in ns_specs.items():
                tau_sym_v = next(
                    (ts for ts, vv in ex_tau if vv == v), None
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
                    cp = cp.subs(num_params)
                try:
                    cp = cp.simplify_full()
                except (ValueError, RuntimeError, AttributeError):
                    pass
        per_td_cp.append(cp)

    # ── Analytic mode-sum precomputation (Stage 4a: grouped polygon /
    # grouped poset).  All td's in this group share the same pole
    # spectrum (same propagator_data); residues differ only via the
    # per-td (ri, pi) edge indices.  When non-local cumulant kernels
    # are present (any vertex_leg_time is non-empty) the integrand
    # contains non-rational factors and the analytic path cannot fire
    # — fall back to scipy.nquad in that case.
    pole_vals = propagator_data.get('pole_vals')
    C_mats   = propagator_data.get('C_mats')
    grouped_analytic_enabled = (
        USE_GROUPED_ANALYTIC_MODESUM
        and pole_vals is not None
        and C_mats is not None
        and not any(vlt for vlt in per_td_vertex_leg_time)
    )
    pred_edges = list(D.edges())
    if grouped_analytic_enabled:
        try:
            from sage.all import CDF as _CDF
            lambdas_per_pole = tuple(
                complex(_CDF(SR(p))) * 1j for p in pole_vals
            )
            n_poles = len(lambdas_per_pole)
            # per-td per-edge per-pole residue tensor.  Indexed by
            # [td_idx][edge_in_pred_edges][pole_idx].  Built once here
            # and reused across subsets.
            residues_per_td_edge = []
            for td in typed_diagrams:
                this_td = []
                for ek in pred_edges:
                    ri, pi = _lookup_prop_indices(td, ek)
                    this_td.append(tuple(
                        complex(_CDF(SR(C_mats[k][pi, ri])))
                        for k in range(n_poles)
                    ))
                residues_per_td_edge.append(this_td)
        except (TypeError, ValueError, KeyError, IndexError):
            grouped_analytic_enabled = False
            lambdas_per_pole = ()
            residues_per_td_edge = []
            n_poles = 0
    else:
        lambdas_per_pole = ()
        residues_per_td_edge = []
        n_poles = 0

    # ── External-time bookkeeping (shared) ───────────────────────
    free_ext_idx = [
        j for j in range(len(ext_time_vars))
        if (origin_leaf_idx is None or j != origin_leaf_idx)
    ]
    free_ext_syms = [ext_time_vars[j] for j in free_ext_idx]

    # All td's in the group have the same edges (same prediagram).
    n_edges = len(per_td_edge_info[0])
    n_subsets_total = 2 ** n_edges

    # ── 2^|E| subset loop, with summation across td's ────────────
    subset_contributions = []
    subset_diagnostics = []

    for subset_bits in range(n_subsets_total):
        delta_edges  = [i for i in range(n_edges)
                        if (subset_bits >> i) & 1]
        smooth_edges = [i for i in range(n_edges)
                        if not ((subset_bits >> i) & 1)]

        # Per-td filter: any td whose δ-coeff is zero on a δ-edge
        # contributes nothing to this subset.  Keep only the td's
        # that pass.
        contributing_td = []
        for td_idx in range(len(typed_diagrams)):
            ei = per_td_edge_info[td_idx]
            zero = any(abs(complex(ei[i]['delta_coeff'])) < 1e-15
                       for i in delta_edges)
            if not zero:
                contributing_td.append(td_idx)
        if not contributing_td:
            continue

        # ── Solve δ-edge equalities using td[0]'s dt_syms ─────────
        # Every td shares the same dt SYMBOLS (they depend only on
        # vertex_time, which is shared).  Different td's have
        # different dt_sym only if their vertex_leg_time maps differ
        # — that happens for noise-source vertices.  Defensive
        # check: if td's disagree on dt_sym for any δ-edge, fall
        # back to per-td handling for this subset.
        ref_ei = per_td_edge_info[contributing_td[0]]
        dt_consistent = True
        for td_idx in contributing_td[1:]:
            td_ei = per_td_edge_info[td_idx]
            for i in delta_edges:
                if SR(td_ei[i]['dt_sym']) != SR(ref_ei[i]['dt_sym']):
                    dt_consistent = False
                    break
            if not dt_consistent:
                break
        if not dt_consistent:
            # Defensive fallback: don't group this subset across
            # td's with mismatching dt structure.  Could be subdivided
            # by dt_sym-equivalence-class, but the heterogeneous
            # Hawkes cases don't hit this branch.
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'dt_sym_mismatch_skipped',
            })
            continue

        substitutions = {}
        remaining_int_vars = list(integration_vars_grouped)
        ext_time_equalities = []
        subset_infeasible = False
        for ei_idx in delta_edges:
            eq_expr = SR(ref_ei[ei_idx]['dt_sym']).subs(substitutions)
            try:
                eq_vars = set(eq_expr.variables())
            except AttributeError:
                eq_vars = set()
            int_var_to_eliminate = None
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
                ext_time_equalities.append(eq_expr)
        if subset_infeasible:
            continue

        # Shot-noise δ contributions (residual ext-time equalities):
        # for the prototype, skip these so we don't have to enumerate
        # δ-spike contributions across the group.  These show up only
        # at τ=0 for same-population legs in the heterogeneous Hawkes
        # cases, and the per-diagram path also marks them as separate
        # ``delta_contributions``.  Re-enable when needed.
        has_shotnoise = False
        for eq in ext_time_equalities:
            try:
                if bool(eq.is_zero()):
                    continue
            except Exception:
                pass
            has_shotnoise = True
            break
        if has_shotnoise:
            subset_diagnostics.append({
                'delta_edges': delta_edges,
                'smooth_edges': smooth_edges,
                'status': 'shotnoise_skipped_in_prototype',
            })
            continue

        # ── Build the SUMMED subset factor across contributing td's ─
        subset_factor_group = SR(0)
        for td_idx in contributing_td:
            ei = per_td_edge_info[td_idx]
            cp = per_td_cp[td_idx]
            term = cp
            for i in delta_edges:
                term = term * SR(ei[i]['delta_coeff'])
            for i in smooth_edges:
                term = term * ei[i]['smooth_factor']
            term = term.subs(substitutions)
            subset_factor_group = subset_factor_group + term

        try:
            subset_factor_group = subset_factor_group.expand()
        except Exception:
            pass

        # Trivially-zero subset → skip.
        try:
            if subset_factor_group.is_trivial_zero():
                continue
        except AttributeError:
            if str(subset_factor_group) == '0':
                continue

        # Smooth-edge retardation constraints (shared — built from
        # the reference edge_info's dt_sym since they're identical
        # across contributing td's at this point).
        subset_retard = []
        for ei_idx in smooth_edges:
            c = SR(ref_ei[ei_idx]['dt_sym']).subs(substitutions)
            subset_retard.append(c)

        m_sub = len(remaining_int_vars)
        fc_vars_sub = list(remaining_int_vars) + list(free_ext_syms)

        # Sanity: the summed integrand should have only expected vars.
        try:
            subset_free_vars = set(subset_factor_group.variables())
        except AttributeError:
            subset_free_vars = set()
        unexpected = subset_free_vars - set(fc_vars_sub)
        if unexpected:
            return {
                'status': 'failed', 'contribution': None,
                'n_diagrams': len(typed_diagrams),
                'integration_vars': integration_vars,
                'reason': (
                    f"[subset {bin(subset_bits)}] grouped stripped "
                    f"integrand contains unexpected free symbols "
                    f"{unexpected}; pass them via num_params."
                ),
            }

        try:
            integrand_fc = fast_callable(
                subset_factor_group,
                vars=fc_vars_sub, domain=CDF,
            )
        except Exception as exc:
            return {
                'status': 'failed', 'contribution': None,
                'n_diagrams': len(typed_diagrams),
                'integration_vars': integration_vars,
                'reason': (
                    f"[subset {bin(subset_bits)}] grouped "
                    f"fast_callable failed: {exc}"
                ),
            }

        # Polytope linear-coefficient extraction.
        subset_constraint_data = []
        constraint_err = None
        for c in subset_retard:
            c_sr = SR(c)
            try:
                a_int = [float(c_sr.coefficient(v))
                         for v in remaining_int_vars]
                a_ext = [float(c_sr.coefficient(s))
                         for s in free_ext_syms]
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
                'status': 'failed', 'contribution': None,
                'n_diagrams': len(typed_diagrams),
                'integration_vars': integration_vars,
                'reason': (
                    f"[subset {bin(subset_bits)}] constraint not "
                    f"linear: {constraint_err}"
                ),
            }

        # τ_v finite caps for non-local cumulant kernels (rare in
        # heterogeneous Hawkes; harmless when absent).
        if grouped_extra_tau and remaining_int_vars:
            n_iv = len(remaining_int_vars)
            n_ext = len(free_ext_syms)
            for (tau_s, _v) in grouped_extra_tau:
                if tau_s not in remaining_int_vars:
                    continue
                idx = remaining_int_vars.index(tau_s)
                a_up = [0.0] * n_iv
                a_up[idx] = -1.0
                subset_constraint_data.append(
                    (a_up, [0.0] * n_ext, TAU_KERNEL_CAP)
                )
                a_lo = [0.0] * n_iv
                a_lo[idx] = +1.0
                subset_constraint_data.append(
                    (a_lo, [0.0] * n_ext, TAU_KERNEL_CAP)
                )

        # ── Analytic mode-sum path (Stage 4a grouped polygon / poset).
        # Builds the merged-residue pole-tuple list
        #   B_α = Σ_td (cp_td · Π_δedge δ_coeff_td) · Π_smooth_edge C^{(td)}_{α_e, e}
        # and routes through the existing analytic evaluators
        # (``_integrate_2d_polygon_modesum`` for m=2,
        # ``_integrate_nd_polytope_poset_modesum`` for m≥3).  On any
        # construction failure or analytic ``None`` return, falls
        # through to the scipy.nquad path on the summed integrand.
        grouped_pole_tuples = None
        grouped_dummy_modes = None
        # The analytic merged-residue path applies for m ∈ {0, 1, 2, ≥3}.
        # m=2 routes through the polygon evaluator, m≥3 through the
        # causal-poset evaluator, m=0/1 through dedicated grouped
        # closed-form helpers above.  All four share the same
        # ``(B_α, lambdas)`` pole-tuple construction.
        if (grouped_analytic_enabled
                and contributing_td and not grouped_extra_tau
                and (
                    m_sub in (0, 1)
                    or (_fi_mod.USE_POLYGON_M2_INTEGRATOR and m_sub == 2)
                    or (_fi_mod.USE_POSET_INTEGRATOR and m_sub >= 3)
                )):
            try:
                from sage.all import CDF as _CDF
                # Per-contributing-td merged prefactor (cp_td · Π δ_coeffs).
                merged_pf_per_j = []
                for td_idx in contributing_td:
                    ei = per_td_edge_info[td_idx]
                    cp = per_td_cp[td_idx]
                    merged = complex(_CDF(SR(cp)))
                    for di in delta_edges:
                        merged *= complex(
                            _CDF(SR(ei[di]['delta_coeff']))
                        )
                    merged_pf_per_j.append(merged)
                # Per-contributing-td smooth-edge residue arrays.
                smooth_residues_per_j = [
                    tuple(
                        residues_per_td_edge[td_idx][ee]
                        for ee in smooth_edges
                    )
                    for td_idx in contributing_td
                ]
                n_smooth = len(smooth_edges)
                # Enumerate multi-indices over per-edge poles and sum
                # residues across td's.
                grouped_pole_tuples = []
                idx_alpha = [0] * n_smooth
                while True:
                    B_alpha = 0.0 + 0.0j
                    for j in range(len(merged_pf_per_j)):
                        prod = merged_pf_per_j[j]
                        for ee in range(n_smooth):
                            prod *= smooth_residues_per_j[j][ee][idx_alpha[ee]]
                        B_alpha += prod
                    lambdas = tuple(
                        lambdas_per_pole[idx_alpha[ee]]
                        for ee in range(n_smooth)
                    )
                    grouped_pole_tuples.append((B_alpha, lambdas))
                    # Advance multi-index.
                    ee = 0
                    while ee < n_smooth:
                        idx_alpha[ee] += 1
                        if idx_alpha[ee] < n_poles:
                            break
                        idx_alpha[ee] = 0
                        ee += 1
                    else:
                        break
                # smooth_edge_modes used only for n_smooth check inside
                # the analytic evaluators; dummy with td[0]'s residues.
                first_j = 0
                grouped_dummy_modes = tuple(
                    EdgeModeSum(
                        ri=0, pi=0, delta_coeff=0.0 + 0.0j,
                        modes=tuple(zip(
                            smooth_residues_per_j[first_j][ee],
                            lambdas_per_pole,
                        )),
                    )
                    for ee in range(n_smooth)
                )
            except (TypeError, ValueError):
                grouped_pole_tuples = None
                grouped_dummy_modes = None

        # Build this subset's contribution closure.
        def _make_subset_contrib(fc, cdata, m_val,
                                  pole_tuples=None, dummy_modes=None):
            def _contrib(free_vals):
                # Analytic merged-residue path first.
                if pole_tuples is not None:
                    if m_val == 0:
                        val = _evaluate_grouped_m0_modesum(
                            pole_tuples=pole_tuples,
                            subset_constraint_data=cdata,
                            free_ext_vals=free_vals,
                        )
                        if val is not None:
                            return val
                    elif m_val == 1:
                        val = _integrate_grouped_m1_modesum(
                            pole_tuples=pole_tuples,
                            subset_constraint_data=cdata,
                            free_ext_vals=free_vals,
                            bbox_cap=_fi_mod.POLYGON_BBOX_CAP,
                        )
                        if val is not None:
                            return val
                    elif m_val == 2 and dummy_modes is not None:
                        val = _integrate_2d_polygon_modesum(
                            smooth_edge_modes=list(dummy_modes),
                            prefactor_complex=1.0 + 0.0j,
                            subset_constraint_data=cdata,
                            free_ext_vals=free_vals,
                            pole_tuples=pole_tuples,
                        )
                        if val is not None:
                            return val
                    elif m_val >= 3 and dummy_modes is not None:
                        val = _integrate_nd_polytope_poset_modesum(
                            smooth_edge_modes=list(dummy_modes),
                            prefactor_complex=1.0 + 0.0j,
                            subset_constraint_data=cdata,
                            free_ext_vals=free_vals,
                            m=m_val,
                            pole_tuples=pole_tuples,
                        )
                        if val is not None:
                            return val
                # scipy.nquad fallback on the SR-summed integrand.
                resolved = []
                for (a_int, a_ext, c0) in cdata:
                    c_eff = c0 + sum(a_ext[i] * free_vals[i]
                                     for i in range(len(a_ext)))
                    resolved.append((list(a_int), c_eff))
                return _integrate_polytope(fc, resolved, free_vals, m_val)
            return _contrib

        subset_contributions.append(
            _make_subset_contrib(
                integrand_fc, subset_constraint_data, m_sub,
                pole_tuples=grouped_pole_tuples,
                dummy_modes=grouped_dummy_modes,
            )
        )
        subset_diagnostics.append({
            'delta_edges': delta_edges,
            'smooth_edges': smooth_edges,
            'status': 'evaluated',
            'm_after_delta': m_sub,
            'n_contributing_td': len(contributing_td),
            'analytic_modesum': grouped_pole_tuples is not None,
        })

    # ── Final contribution callable (Wick contraction sum) ────────
    _m0 = _all_mappings[0]
    _m0_inv = {v: k for k, v in _m0.items()}
    _k = len(ext_time_vars)

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
                f"contribution() expects {len(ext_time_vars)} "
                f"positional arguments; got {len(ext_time_values)}."
            )
        total = 0.0 + 0.0j
        for perm in _perms:
            permuted = [ext_time_values[perm[j]] for j in range(_k)]
            free_vals = [float(permuted[j]) for j in free_ext_idx]
            for cfn in subset_contributions:
                total = total + complex(cfn(free_vals))
        return total / _comp

    return {
        'status': 'ok',
        'contribution': contribution,
        'n_diagrams': len(typed_diagrams),
        'integration_vars': integration_vars,
        'loop_number': loop_number,
        'subset_diagnostics': subset_diagnostics,
    }

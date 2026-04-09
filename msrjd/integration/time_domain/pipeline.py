"""
msrjd.integration.time_domain.pipeline
======================================
Phase J orchestrator.

Dispatches each kernel group in a pre-computed `kernel_groups` list
(from `group_diagrams_by_kernel` in the Phase I frequency backend) to
the appropriate Phase J evaluator.

MVP dispatch rule
-----------------
- `loop_number == 0` — bypass Phases 3-5 (kernel reduction, kernel
  caching, parent contraction) and evaluate the group directly via
  `integrate_tree_diagram` (Phase 6 only). The tree evaluator uses
  explicit numerical quadrature (scipy.integrate.quad / nquad) on a
  `fast_callable` version of the integrand, with polytope bounds
  extracted from the retarded Heaviside factors; it returns a Python
  callable `f(*ext_time_values) -> complex`.
- `loop_number > 0` — not in MVP scope. The group is marked `'skipped'`
  and added to `skipped_kernel_ids`; the caller should fall back to
  Phase I's residue backend for those groups.

`total_C` in the returned dict is itself a Python callable that sums
every tree-group contribution at a given external-time point. Pass
`k` positional arguments (one per external time) to evaluate it; if
any group was called with a pinned origin, the value at that position
is ignored.
"""

from sage.all import SR

from msrjd.integration.time_domain.final_integral import (
    integrate_tree_diagram,
)


def compute_correction_td(
    kernel_groups,
    propagator_data,
    k,
    num_params=None,
    ext_time_vars=None,
    origin_leaf_idx=0,
    timeout_sec=30,
):
    r"""
    Phase J entry point: evaluate a set of kernel groups in the time
    domain via explicit numerical quadrature.

    Dispatch per kernel group:
    - `loop_number == 0` → bypass Phases 3-5 and call
      `integrate_tree_diagram` on the group's representative directly
      (Phase 6 only).
    - `loop_number > 0` → mark as `'skipped'` with a reason and add to
      `skipped_kernel_ids` so the caller can fall back to Phase I.

    Parameters
    ----------
    kernel_groups : list of dict
        As returned by `msrjd.integration.symbolic.group_diagrams_by_kernel`.
    propagator_data : dict
        Standard pipeline propagator dict; must contain 'pole_vals' and
        'C_mats' for the time-domain path.
    k : int
        Number of external legs in the correlator.
    num_params : dict or None
        Numerical parameter substitutions. Passed through to the tree
        evaluator so the propagator matrix and combined prefactor are
        built numerically. Required for any diagram whose propagator
        or prefactor contains free symbolic parameters.
    ext_time_vars : list of SR or None
        External time variables (one per external leg). If None,
        defaults to `[t_1, t_2, ..., t_k]`. These symbols show up
        only inside `integrate_tree_diagram` for coefficient
        extraction and polytope parameterization — the returned
        callable takes numeric values, not symbolic.
    origin_leaf_idx : int or None
        Which external leaf's time to pin to zero. Default 0. Pass None
        to leave all external times free.
    timeout_sec : int
        Unused on the numerical path; kept for API compatibility with
        earlier symbolic-integration builds.

    Returns
    -------
    dict with keys:
        'total_C' : callable
            `f(*ext_time_values) -> complex`. Sums the contributions of
            every tree kernel group that was successfully evaluated.
            Returns `0+0j` when there are no evaluated groups.
        'groups' : list of dict
            Per-group diagnostics with keys:
              - 'kernel_id'       : the group signature (tuple)
              - 'loop_number'     : int
              - 'n_diagrams'      : int
              - 'handled_by'      : 'tree_evaluator' | 'skipped'
              - 'reason'          : str
              - 'representation'  : 'numerical' | None
              - 'contribution'    : callable or None
                (Contract: `contribution` must be a Python callable
                `f(*ext_time_values) -> complex`. The numerical tree
                evaluator always returns a callable; later hybrid
                kernels — polyhedral-exponential, scipy.quad on a
                reduced kernel, etc. — will do the same.)
        'skipped_kernel_ids' : list of tuple
            Kernel signatures that were NOT evaluated by Phase J — the
            caller should fall back to Phase I for these.
        'ext_time_vars' : list of SR
            The external time variables used internally during
            coefficient extraction.
    """
    if ext_time_vars is None:
        ext_time_vars = [
            SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
            for j in range(k)
        ]

    tree_callables = []
    groups_out = []
    skipped = []
    all_delta_contributions = []  # shot-noise δ spikes (see final_integral)

    for g in kernel_groups:
        loop_number = g.get('loop_number', 0)
        signature = g.get('signature')
        n_diagrams = g.get('n_diagrams', len(g.get('diagrams', [])))
        combined_prefactor = g.get('combined_prefactor')
        representative_ir = g.get('representative_ir')
        diagrams = g.get('diagrams', [])

        if not diagrams:
            groups_out.append({
                'kernel_id': signature,
                'loop_number': loop_number,
                'n_diagrams': 0,
                'handled_by': 'skipped',
                'reason': 'empty group',
                'representation': None,
                'contribution': None,
            })
            skipped.append(signature)
            continue

        if loop_number == 0:
            td = diagrams[0]
            result = integrate_tree_diagram(
                typed_diagram=td,
                representative_ir=representative_ir,
                propagator_data=propagator_data,
                combined_prefactor=combined_prefactor,
                ext_time_vars=ext_time_vars,
                num_params=num_params,
                origin_leaf_idx=origin_leaf_idx,
                timeout_sec=timeout_sec,
            )
            if result['status'] == 'ok':
                contribution = result['contribution']
                tree_callables.append(contribution)
                # Aggregate any shot-noise δ spikes the tree evaluator
                # produced for this group. Each is a structured dict
                # that the caller can feed to
                # `eval_delta_contributions_on_tau_grid` to add a
                # discrete spike to a τ grid.
                all_delta_contributions.extend(
                    result.get('delta_contributions', [])
                )
                groups_out.append({
                    'kernel_id': signature,
                    'loop_number': loop_number,
                    'n_diagrams': n_diagrams,
                    'handled_by': 'tree_evaluator',
                    'reason': '',
                    'representation': 'numerical',
                    'contribution': contribution,
                    'n_delta_contributions': len(
                        result.get('delta_contributions', [])
                    ),
                })
            else:
                groups_out.append({
                    'kernel_id': signature,
                    'loop_number': loop_number,
                    'n_diagrams': n_diagrams,
                    'handled_by': 'skipped',
                    'reason': (
                        f"tree evaluator status={result['status']}"
                        + (f" ({result['reason']})"
                           if 'reason' in result else '')
                    ),
                    'representation': None,
                    'contribution': None,
                })
                skipped.append(signature)
        else:
            groups_out.append({
                'kernel_id': signature,
                'loop_number': loop_number,
                'n_diagrams': n_diagrams,
                'handled_by': 'skipped',
                'reason': f'loop_number = {loop_number}: not in MVP',
                'representation': None,
                'contribution': None,
            })
            skipped.append(signature)

    # Build the combined callable: sum of all evaluated tree group
    # contributions. Each group's callable takes k positional arguments
    # (one per ext_time_var); total_C has the same signature.
    def total_C(*ext_time_values):
        if not tree_callables:
            return 0.0 + 0.0j
        total = 0.0 + 0.0j
        for fn in tree_callables:
            val = fn(*ext_time_values)
            total = total + complex(val)
        return total

    return {
        'total_C': total_C,
        'delta_contributions': all_delta_contributions,
        'groups': groups_out,
        'skipped_kernel_ids': skipped,
        'ext_time_vars': ext_time_vars,
    }

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
  `integrate_tree_diagram` (Phase 6 only).
- `loop_number > 0` — not in MVP scope. The group is marked `'skipped'`
  and added to `skipped_kernel_ids`; the caller should fall back to
  Phase I's residue backend for those groups.

This lets the MVP validate the full Phase J evaluation layer
(time-domain propagator extraction, vertex-time integration, direction
conventions, translation fixing, dispatch) on tree-level diagrams
WITHOUT needing the kernel-reduction / contraction machinery that
Extension 1 will build.
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
    domain.

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
        Numerical parameter substitutions; passed through to the tree
        evaluator so the propagator matrix is built numerically.
    ext_time_vars : list of SR or None
        External time variables (one per external leg). If None,
        defaults to `[t_1, t_2, ..., t_k]` matching the Phase I
        convention in `build_integrand_stationary`.
    origin_leaf_idx : int or None
        Which external leaf's time to pin to zero. Default 0. Pass None
        to leave all external times free.
    timeout_sec : int
        Per-integration wall-clock timeout (see
        `integrate_tree_diagram`).

    Returns
    -------
    dict with keys:
        'total_C' : SR expression
            Sum of Phase J contributions across all kernel groups that
            were handled (i.e., tree-level groups in the MVP).
        'groups' : list of dict
            Per-group diagnostics with keys:
              - 'kernel_id' : the group signature (tuple)
              - 'loop_number' : int
              - 'n_diagrams' : int
              - 'handled_by' : 'tree_evaluator' | 'skipped'
              - 'reason' : str
              - 'representation' : 'symbolic' | 'numerical' | None
              - 'contribution' : SR expression or None
                (Contract: evaluable as a function of external times —
                either a Sage SR expression in the external time
                symbols, or a Python callable f(*ext_times). In the MVP
                the tree evaluator returns SR, but later hybrid kernels
                may return callable-only, and downstream plotting code
                should inspect 'representation' and dispatch accordingly.)
        'skipped_kernel_ids' : list of tuple
            Kernel signatures that were NOT evaluated by Phase J — the
            caller should fall back to Phase I for these.
        'ext_time_vars' : list of SR
            The external time variables the result is expressed in.
    """
    if ext_time_vars is None:
        ext_time_vars = [
            SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
            for j in range(k)
        ]

    total_C = SR(0)
    groups_out = []
    skipped = []

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
                total_C = total_C + contribution
                groups_out.append({
                    'kernel_id': signature,
                    'loop_number': loop_number,
                    'n_diagrams': n_diagrams,
                    'handled_by': 'tree_evaluator',
                    'reason': '',
                    'representation': 'symbolic',
                    'contribution': contribution,
                })
            else:
                groups_out.append({
                    'kernel_id': signature,
                    'loop_number': loop_number,
                    'n_diagrams': n_diagrams,
                    'handled_by': 'skipped',
                    'reason': (
                        f'tree evaluator returned status '
                        f"'{result['status']}'"
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

    return {
        'total_C': total_C,
        'groups': groups_out,
        'skipped_kernel_ids': skipped,
        'ext_time_vars': ext_time_vars,
    }

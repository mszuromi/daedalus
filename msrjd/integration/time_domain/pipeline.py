"""
msrjd.integration.time_domain.pipeline
======================================
Time-domain tree-level correlator evaluation.

Evaluates each typed diagram directly in the time domain via explicit
numerical quadrature of vertex-time integrals.  No frequency-domain
integral construction or loop-kernel grouping is needed — the pipeline
works directly from:

  1. A list of typed diagrams (from the enumeration pipeline)
  2. Their scalar prefactors (from ``classify_coefficient_factors``)
  3. The retarded propagator in pole-residue form (``pole_vals``,
     ``C_mats``, ``D_delta``)

For tree-level diagrams (loop_number == 0), each diagram is evaluated
independently via ``integrate_tree_diagram``.  Loop diagrams
(loop_number > 0) are marked as skipped — loop-kernel reduction is
deferred to a future extension.

Convention
----------
``total_C(t_1, t_2, ..., t_k)`` returns the sum of all tree-level
diagram contributions.  Position i is ALWAYS the time of
``external_fields[i]``:

  - ``external_fields[0]`` → ``t_1`` (base time, pinned to 0)
  - ``external_fields[1]`` → ``t_2``, ``τ_1 = t_2 - t_1``
  - ``external_fields[2]`` → ``t_3``, ``τ_2 = t_3 - t_1``

This is enforced by the canonical time remapping in
``integrate_tree_diagram``.
"""

from sage.all import SR

from msrjd.integration.time_domain.final_integral import (
    integrate_diagram,
    _loop_number_from_graph,
)


def compute_correction_td(
    typed_diagrams=None,
    prefactors=None,
    propagator_data=None,
    k=None,
    num_params=None,
    ext_time_vars=None,
    origin_leaf_idx=0,
    external_fields=None,
    # Legacy support: accept kernel_groups as first arg
    kernel_groups=None,
):
    r"""
    Time-domain entry point: evaluate typed diagrams via explicit
    numerical quadrature of vertex-time integrals.

    Accepts EITHER:
    - ``typed_diagrams`` + ``prefactors``: direct diagram list (preferred)
    - ``kernel_groups``: legacy format from ``group_diagrams_by_kernel``

    For each tree-level diagram, calls ``integrate_tree_diagram``
    directly.  Loop diagrams are marked as skipped.

    Parameters
    ----------
    typed_diagrams : list of TypedDiagram or None
        The enumerated, deduplicated typed diagrams.  Each diagram is
        evaluated independently (no kernel grouping needed at tree level).
    prefactors : list of SR/numeric or None
        Scalar prefactor for each diagram (from
        ``classify_coefficient_factors``).  Must be same length as
        ``typed_diagrams``.
    propagator_data : dict
        Must contain ``'pole_vals'``, ``'C_mats'``, and optionally
        ``'D_delta'``.
    k : int
        Number of external legs.
    num_params : dict or None
        Numerical parameter substitutions.
    ext_time_vars : list of SR or None
        External time symbols in canonical order.  Defaults to
        ``[t_1, ..., t_k]``.
    origin_leaf_idx : int or None
        Which canonical position to pin to zero.  Default 0.
    external_fields : list of tuple or None
        Canonical external field list, e.g. ``[('dn',1), ('dn',1), ('dn',2)]``.

    Returns
    -------
    dict with keys:
        'total_C' : callable
            ``f(*ext_time_values) -> complex``.  Position i = time of
            ``external_fields[i]``.
        'delta_contributions' : list of dict
            Surviving delta contributions (distributional, not added to
            ``total_C``).
        'groups' : list of dict
            Per-diagram diagnostics.
        'skipped_kernel_ids' : list
            Diagrams not evaluated (loop_number > 0).
        'ext_time_vars' : list of SR
    """
    # ── Legacy support: unpack kernel_groups format ──
    if typed_diagrams is None and kernel_groups is not None:
        typed_diagrams = []
        prefactors = []
        for g in kernel_groups:
            for td in g.get('diagrams', []):
                typed_diagrams.append(td)
                prefactors.append(g.get('combined_prefactor'))
    elif typed_diagrams is None:
        typed_diagrams = []
        prefactors = []

    if ext_time_vars is None:
        ext_time_vars = [
            SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
            for j in range(k)
        ]

    tree_callables = []
    groups_out = []
    skipped = []
    all_delta_contributions = []

    for idx, (td, pf) in enumerate(zip(typed_diagrams, prefactors)):
        loop_number = _loop_number_from_graph(td)

        # ``integrate_diagram`` handles every loop order -- tree and
        # 1-loop share the same vertex-time integration algorithm, and
        # the DAG structure our enumerator produces keeps the polytope
        # feasible even when the underlying undirected graph has cycles
        # (multi-edges between the same vertex pair just contribute
        # duplicated Heaviside constraints, which are redundant but
        # harmless).  Higher loop orders pass through as well, but for
        # ell >= 2 the 2^|E| delta-subset sum grows quickly; monitor
        # runtime and consider per-diagram timeouts if needed.
        result = integrate_diagram(
            typed_diagram=td,
            propagator_data=propagator_data,
            combined_prefactor=pf,
            ext_time_vars=ext_time_vars,
            num_params=num_params,
            origin_leaf_idx=origin_leaf_idx,
            external_fields=external_fields,
        )
        if result['status'] == 'ok':
            contribution = result['contribution']
            tree_callables.append(contribution)
            all_delta_contributions.extend(
                result.get('delta_contributions', [])
            )
            groups_out.append({
                'kernel_id': idx,
                'loop_number': loop_number,
                'n_diagrams': 1,
                'handled_by': (
                    'tree_evaluator' if loop_number == 0
                    else 'loop_evaluator'
                ),
                'reason': '',
                'representation': 'numerical',
                'contribution': contribution,
                'n_delta_contributions': len(
                    result.get('delta_contributions', [])
                ),
            })
        else:
            groups_out.append({
                'kernel_id': idx,
                'loop_number': loop_number,
                'n_diagrams': 1,
                'handled_by': 'skipped',
                'reason': (
                    f"evaluator status={result['status']}"
                    + (f" ({result['reason']})"
                       if 'reason' in result else '')
                ),
                'representation': None,
                'contribution': None,
            })
            skipped.append(idx)

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

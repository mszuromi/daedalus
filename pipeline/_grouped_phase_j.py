"""
pipeline._grouped_phase_j
=========================
Prototype prediagram-grouped Phase J wrapper.

Mirrors the per-diagram loop in ``compute.py``'s step [7/7]:

    for ell in sorted(unique_by_ell):
        td_result = compute_correction_td(typed_diagrams=…, prefactors=…)

but groups the typed diagrams of each ell by their parent prediagram
identity and dispatches each group to ``integrate_grouped_diagram``,
which sums the integrands BEFORE the per-subset fast_callable +
quadrature.  See ``msrjd/integration/time_domain/grouped_integral.py``
for the math.

API
---
``compute_correction_td_grouped`` returns the SAME dict shape as
``compute_correction_td`` (``total_C`` callable, ``total_C_batch``,
``ext_time_vars``), so it can be swapped behind a single flag in
``compute.py`` without touching the τ-grid evaluation or saving
code.

Status
------
Prototype.  Not wired into ``compute_cumulants`` by default — use
``use_grouped_phase_j=True`` to opt in.  See
``compute.py`` for the dispatch.
"""

from collections import defaultdict

from sage.all import SR

from msrjd.integration.time_domain.grouped_integral import (
    integrate_grouped_diagram,
)


def compute_correction_td_grouped(
    typed_diagrams,
    prefactors,
    propagator_data,
    k,
    num_params=None,
    ext_time_vars=None,
    origin_leaf_idx=0,
    external_fields=None,
):
    """Group typed diagrams by parent prediagram, then sum-and-integrate.

    Same call signature and return shape as
    ``msrjd.integration.time_domain.compute_correction_td``, but with
    the per-diagram inner loop replaced by a per-prediagram inner loop
    that hands each group to ``integrate_grouped_diagram``.
    """
    if ext_time_vars is None:
        ext_time_vars = [
            SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
            for j in range(k)
        ]

    # ── Group typed_diagrams by parent prediagram + external_legs ──
    # Use ``(id(td.prediagram), external_legs_signature)`` as the key.
    # Subdividing by external_legs is REQUIRED because the Wick
    # contraction enumeration inside ``integrate_grouped_diagram``
    # assumes every td in the group has the same leaf→external-field
    # mapping.  Two typed diagrams that differ only in which leaf
    # carries which external field have different ``_all_mappings``
    # / ``_compensation``, so they can't share a sum.  Per-diagram
    # path handles each independently; the grouped path subdivides.
    def _ext_legs_key(td):
        # Hashable tuple of (leaf, (field_base, pop_idx)) entries.
        # Use leaf id() to avoid relying on leaf being directly
        # comparable across diagrams (it's typically an int, so this
        # is just defensive).
        return tuple(sorted(
            (lf, fld) for lf, fld in td.external_legs.items()
        ))

    groups = defaultdict(lambda: {'tds': [], 'cps': []})
    for td, pf in zip(typed_diagrams, prefactors):
        key = (id(td.prediagram), _ext_legs_key(td))
        groups[key]['tds'].append(td)
        groups[key]['cps'].append(pf)

    group_callables = []
    groups_out = []
    skipped = []

    for group_idx, (key, payload) in enumerate(groups.items()):
        tds = payload['tds']
        cps = payload['cps']
        result = integrate_grouped_diagram(
            typed_diagrams=tds,
            combined_prefactors=cps,
            propagator_data=propagator_data,
            ext_time_vars=ext_time_vars,
            num_params=num_params,
            origin_leaf_idx=origin_leaf_idx,
            external_fields=external_fields,
        )
        if result['status'] == 'ok':
            group_callables.append(result['contribution'])
            groups_out.append({
                'kernel_id':   group_idx,
                'loop_number': result.get('loop_number', 0),
                'n_diagrams':  result['n_diagrams'],
                'handled_by':  'grouped_evaluator',
                'reason':      None,
                'representation': None,
            })
        else:
            skipped.append({
                'group_idx':  group_idx,
                'n_diagrams': result['n_diagrams'],
                'reason':     result.get('reason', '<unknown>'),
            })
            groups_out.append({
                'kernel_id':   group_idx,
                'loop_number': 0,
                'n_diagrams':  result['n_diagrams'],
                'handled_by':  'skipped',
                'reason':      result.get('reason', '<unknown>'),
            })

    # ── Build total_C and total_C_batch ──────────────────────────
    def total_C(*ext_time_values):
        total = 0.0 + 0.0j
        for fn in group_callables:
            total = total + complex(fn(*ext_time_values))
        return total

    def total_C_batch(tau_points, parallel=False, n_workers=None):
        # Match compute_correction_td's API.  For the prototype, just
        # serially evaluate (parallelism can be plumbed in once the
        # baseline correctness is confirmed).
        return [complex(total_C(*pt)) for pt in tau_points]

    return {
        'total_C':            total_C,
        'total_C_batch':      total_C_batch,
        'eval_per_diagram_batch': None,    # not supported in prototype
        'delta_contributions': [],          # not supported in prototype
        'groups':             groups_out,
        'skipped_kernel_ids': skipped,
        'ext_time_vars':      ext_time_vars,
    }

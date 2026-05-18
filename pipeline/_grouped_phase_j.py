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
from msrjd.integration.time_domain.pipeline import compute_correction_td


# Subset-level statuses inside ``integrate_grouped_diagram``'s
# ``subset_diagnostics`` that mean "the grouped path didn't actually
# integrate this subset, it dropped it."  Any group hitting one of these
# at the SUBSET level still returns ``status: 'ok'`` overall but with a
# silently-incomplete integrand — that's the production bug we're fixing.
# We list the known skip reasons explicitly so a new skip status added
# later doesn't silently regress us.
_SUBSET_SKIP_STATUSES = frozenset({
    'shotnoise_skipped_in_prototype',
    'dt_sym_mismatch_skipped',
})


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
    fallback_to_perdiag = []   # groups where we fell back to per-diag

    def _perdiag_fallback(tds, cps):
        """Compute this group via the per-diag path (compute_correction_td).
        Returns ``(contribution_callable, n_diagrams)``.  Wrapping it here
        keeps the fallback isolated from the grouped path's bookkeeping —
        the returned callable has the same ``(*ext_time_values) -> complex``
        signature as ``integrate_grouped_diagram``'s ``contribution``.
        """
        pd_result = compute_correction_td(
            typed_diagrams=tds,
            prefactors=cps,
            k=k,
            propagator_data=propagator_data,
            external_fields=external_fields,
            num_params=num_params,
            origin_leaf_idx=origin_leaf_idx,
        )
        return pd_result['total_C'], len(tds)

    for group_idx, (key, payload) in enumerate(groups.items()):
        tds = payload['tds']
        cps = payload['cps']

        # Tier 1: try the grouped evaluator.  Even on ``status='ok'``,
        # inspect subset_diagnostics for the prototype-skip statuses —
        # those silently drop subsets from the integrand and produce a
        # numerically wrong group total (as proven by the isolation test
        # at spike-reset k=1 ell=2: 5/7 multi-diagram groups dropped to 0).
        result = integrate_grouped_diagram(
            typed_diagrams=tds,
            combined_prefactors=cps,
            propagator_data=propagator_data,
            ext_time_vars=ext_time_vars,
            num_params=num_params,
            origin_leaf_idx=origin_leaf_idx,
            external_fields=external_fields,
        )

        # Detect the silent-skip case: any subset whose status is one
        # of the prototype-skip codes means the grouped integrand is
        # incomplete and we MUST fall back.
        has_skipped_subset = False
        if result.get('status') == 'ok':
            for diag in (result.get('subset_diagnostics') or []):
                if diag.get('status') in _SUBSET_SKIP_STATUSES:
                    has_skipped_subset = True
                    break

        if result.get('status') == 'ok' and not has_skipped_subset:
            # All subsets evaluated by the grouped path — accept it.
            group_callables.append(result['contribution'])
            groups_out.append({
                'kernel_id':   group_idx,
                'loop_number': result.get('loop_number', 0),
                'n_diagrams':  result['n_diagrams'],
                'handled_by':  'grouped_evaluator',
                'reason':      None,
                'representation': None,
            })
        elif result.get('status') == 'ok' and has_skipped_subset:
            # Tier 2 fallback: grouped left at least one subset silently
            # un-integrated.  Re-compute the entire group via per-diag so
            # the total is numerically correct.  This trades the grouped
            # speedup on this group for correctness; it's the same
            # behaviour the caller would get with use_grouped_phase_j=False
            # on this group only.
            fb_fn, n_tds = _perdiag_fallback(tds, cps)
            group_callables.append(fb_fn)
            skip_reasons = sorted({
                diag.get('status')
                for diag in (result.get('subset_diagnostics') or [])
                if diag.get('status') in _SUBSET_SKIP_STATUSES
            })
            fallback_to_perdiag.append({
                'group_idx':       group_idx,
                'n_diagrams':      n_tds,
                'subset_skip_reasons': skip_reasons,
            })
            groups_out.append({
                'kernel_id':   group_idx,
                'loop_number': result.get('loop_number', 0),
                'n_diagrams':  n_tds,
                'handled_by':  'perdiag_fallback',
                'reason':      'subset_skip:' + '+'.join(skip_reasons),
                'representation': None,
            })
        else:
            # Grouped evaluator failed at the group level (not a subset
            # skip — e.g. early validation).  Fall back to per-diag too
            # rather than silently dropping the whole group.
            try:
                fb_fn, n_tds = _perdiag_fallback(tds, cps)
                group_callables.append(fb_fn)
                groups_out.append({
                    'kernel_id':   group_idx,
                    'loop_number': 0,
                    'n_diagrams':  n_tds,
                    'handled_by':  'perdiag_fallback',
                    'reason':      'grouped_status:' +
                                   str(result.get('reason', '<unknown>')),
                    'representation': None,
                })
                fallback_to_perdiag.append({
                    'group_idx':  group_idx,
                    'n_diagrams': n_tds,
                    'subset_skip_reasons': ['group_level_failure'],
                })
            except Exception as exc:
                # Even per-diag failed.  Now we genuinely have to skip,
                # but record it loudly.
                skipped.append({
                    'group_idx':  group_idx,
                    'n_diagrams': result.get('n_diagrams', len(tds)),
                    'reason':     (f"grouped failed ({result.get('reason')}) "
                                   f"AND per-diag fallback failed ({exc!r})"),
                })
                groups_out.append({
                    'kernel_id':   group_idx,
                    'loop_number': 0,
                    'n_diagrams':  result.get('n_diagrams', len(tds)),
                    'handled_by':  'skipped',
                    'reason':      f'grouped+perdiag both failed: {exc!r}',
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
        'fallback_to_perdiag': fallback_to_perdiag,
        'ext_time_vars':      ext_time_vars,
    }

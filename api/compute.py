"""
api.compute — single-call entry point for the full MSR-JD pipeline.

Wraps the notebook flow (cells 1–28 of hawkes_td_*.ipynb) into a
function that takes a model dict + parameter dict + run config, runs
the entire FT expansion → diagram enumeration → Phase J integration
chain, and returns a result dict (with optional .npz save).
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any

import math
import numpy as np
from sage.all import SR

# engine internals
from engine.core.field_theory import FieldTheory
from engine.core.vertices import (
    extract_vertex_types, extract_source_types, NoiseSourceType,
)
from engine.diagrams.type_assignment import build_field_index_map
from engine.diagrams.symmetry import classify_coefficient_factors
from engine.integration.time_domain.pipeline import (
    compute_correction_td,
)

# Pipeline-package helpers
from api._propagator import build_propagator, compute_poles_and_residues
from api._mean_field import solve_mean_field
from api._mean_field_dae import (
    solve_mean_field_dae_compat,
    linear_stability,
)
from api._diagrams  import enumerate_unique_diagrams
from engine.enumeration import prediagram_cache as _pd_cache
from api.save       import _phase_j_convention, _phase_j_flags
from api.access     import (
    MeanField, Parameters, normalize_external_fields,
)

# Itô equal-time regularizer (magnitude).  Our SDEs use the Itô convention, which
# is LEFT-continuous (Θ(0)=Θ(0⁻)); the equal-time *observable* is therefore the
# LEFT limit.  At a τ=0 grid point we evaluate the correlator at −_ITO_EPS
# (≪ every physical timescale here, O(1)) instead of exactly 0, so a coincident
# causal chamber survives and the spurious single-point τ=0 dip disappears.  (For
# the stationary/symmetric correlators here C is even, so the left limit equals the
# right; the sign only matters for a genuine τ=0 jump, where Itô takes the left.)
_ITO_EPS = 1e-6


def _ito_nudge_callable(fn, k):
    """Wrap a user-facing C(τ) callable so its equal-time evaluation matches the
    Itô LEFT-limit used on the C_tau grid (see ``_ITO_EPS`` and the k=2 grid
    construction below).  Without this, ``total_C(t0, t0)`` would sample the
    measure-zero τ=0 step discontinuity and disagree with ``C_tau`` at τ=0.

    Only the k=2 single-axis slice has a well-defined "anchor leg" (leg 0); for
    other k the callable is returned unchanged (``daedalus.run`` places the
    k≥3 times itself: ``_kpoint_slice_times`` for the slices,
    ``_kpoint_pinned_time`` per leg for the full grid).  Mirrors the grid
    nudge: a second-leg time within 1e-12 of the anchor leg's time is moved
    to ``anchor − _ITO_EPS``.
    """
    if k != 2:
        return fn
    # ``not (<=)`` rather than ``>``: a NaN second time compares False both
    # ways and must reach the engine as NaN (which refuses it), never be
    # moved to the equal-time left limit.
    return lambda t0, t1: fn(t0, t0 - _ITO_EPS if abs(t1 - t0) <= 1e-12 else t1)


def _phase_j_warning_scope():
    """The scope that aggregates the Phase J hardened quadrature fallback's
    warnings over one evaluation (``final_integral.nquad_warning_scope``:
    one warning per model, source and loop order, with counts), or a null
    context for an engine without it."""
    import contextlib
    from engine.integration.time_domain import final_integral as _fi
    scope = getattr(_fi, 'nquad_warning_scope', None)
    return scope() if scope is not None else contextlib.nullcontext()


def _phase_j_scoped(fn):
    """``fn`` with every call inside ``_phase_j_warning_scope``."""
    def scoped(*args):
        with _phase_j_warning_scope():
            return fn(*args)
    return scoped


def _tau_extent(tau_grid, tau_max):
    """The largest |time difference| the run evaluates: the tau grid's
    extent, else ``tau_max`` (bounds the pole-cluster deformation radius)."""
    try:
        if tau_grid is not None and len(tau_grid):
            return float(np.max(np.abs(np.asarray(tau_grid, dtype=float))))
    except (TypeError, ValueError):
        pass
    return float(tau_max) if tau_max else None


def _ss_route_prediction(prop, num_params, vertex_types):
    """S3: the run-level route prediction (``statespace.run_route``) plus
    ``bypass_deformation``: the run is covered by the state-space route and
    the propagator has a pole-cluster deformation, which then is not run
    (the route uses no poles or residues)."""
    raw = os.environ.get('DAEDALUS_PHASE_J_ENGINE')
    if raw is not None and raw.strip().lower() == 'poles':
        return {'engine': 'poles', 'covers_run': False,
                'reason': 'engine_poles', 'bypass_deformation': False}
    from engine.integration.time_domain import statespace as _ssm
    pred = _ssm.run_route(prop, num_params, vertex_types)
    pred['bypass_deformation'] = bool(
        pred['covers_run'] and prop.get('pole_deformation') is not None)
    return pred


def _pole_cluster_preamble(prop, max_ell, ss_route=None):
    """The run's pole-cluster report (``None`` when no two poles are close)
    and the warning for a cluster the pipeline cannot resolve.  S3: when
    the state-space route covers the run (``ss_route['covers_run']``) the
    poles are not used: status ``'bypassed'`` (the pole engine's status in
    ``status_poles``), no warning."""
    from api import _pole_clusters as _pc
    pc = prop.get('pole_clusters')
    if not pc:
        return None
    pc = dict(pc)
    if ss_route is not None and ss_route.get('covers_run'):
        pc['status_poles'] = pc['status']
        pc['reason_poles'] = pc.get('reason')
        pc['status'] = 'bypassed'
        pc['reason'] = ('the state-space Phase J route takes every subset of '
                        'the run; it uses no poles or residues')
        pc['accuracy'] = None
        return pc
    if pc['status'] in ('unresolved', 'disabled'):
        digits = _pc.lost_digits(pc['kappa_max'], max_ell)
        pc['lost_digits'] = digits
        if digits <= _pc.LOST_DIGITS_WARN:
            return pc                     # recorded; the loss stays below 1e-6
        what = ('a repeated pole, which the pole-residue form cannot '
                'represent' if pc.get('repeated_pole') else
                f'near-degenerate poles (cancellation factor '
                f'{pc["kappa_max"]:.3g}, about {digits:.0f} digits lost at '
                f'max_ell={max_ell})')
        _pc.warn(f'the propagator has {what}; the pole-cluster deformation '
                 f'is {pc["status"]}: {pc["reason"]}.  The cumulant may be '
                 f'inaccurate (see result["pole_clusters"]).')
    return pc


class PhaseJIncompleteWarning(UserWarning):
    """``compute_cumulants`` left diagrams out: Phase J could not evaluate
    them, so the cumulant misses their contributions
    (``result['phase_j_incomplete']`` names them and the reason)."""


from engine.integration.time_domain.final_integral import (  # noqa: E402
    PhaseJResourceError)


class PhaseJIncompleteError(RuntimeError):
    """Raised instead of :class:`PhaseJIncompleteWarning` under
    ``policy={'mode': 'strict'}``.  ``result``: the (incomplete) result
    dict; ``failures``: ``{ell: [failure record, ...]}``."""

    def __init__(self, message, result=None, failures=None):
        super().__init__(message)
        self.result = result
        self.failures = failures or {}


def _phase_j_failures(phase_j_by_ell):
    """``{ell: [{'diagram', 'n_diagrams', 'reason'}, ...]}`` for every
    diagram (per-diagram route) or group (grouped route) that Phase J could
    not evaluate and therefore left out of ``C_tau``; ``{}`` when none.
    Reads the ``skipped_kernel_ids`` of each loop order (ints with the
    reason in ``groups`` on the per-diagram route, dicts on the grouped
    one), so every route reports through the same field."""
    out = {}
    for ell, pj in sorted((phase_j_by_ell or {}).items()):
        if not isinstance(pj, dict):
            continue
        reasons = {g.get('kernel_id'): g.get('reason')
                   for g in (pj.get('groups') or [])
                   if g.get('handled_by') == 'skipped'}
        recs = []
        for sk in pj.get('skipped_kernel_ids') or []:
            if isinstance(sk, dict):
                recs.append({
                    'diagram': sk.get('kernel_id', sk.get('group_idx')),
                    'n_diagrams': sk.get('n_diagrams', 1),
                    'reason': str(sk.get('reason', '')),
                })
            else:
                recs.append({'diagram': sk, 'n_diagrams': 1,
                             'reason': str(reasons.get(sk, ''))})
        if recs:
            out[ell] = recs
    return out


def _policy_is_strict(policy):
    mode = (policy.get('mode') if isinstance(policy, dict)
            else getattr(policy, 'mode', None))
    return mode == 'strict'


def _phase_j_failure_message(failures, model_name):
    n = sum(r['n_diagrams'] for recs in failures.values() for r in recs)
    lines = [f'compute_cumulants({model_name!r}): Phase J could not evaluate '
             f'{n} diagram(s); C_tau is missing them (all of a diagram, or '
             f'its part at the pole-deformation nodes where it failed; '
             f'result["status"] = "incomplete", details in '
             f'result["phase_j_incomplete"]):']
    for ell, recs in failures.items():
        by_reason = {}
        for r in recs:
            by_reason.setdefault(r['reason'], []).append(r['diagram'])
        for reason, ids in by_reason.items():
            shown = ', '.join(map(str, ids[:12])) + (
                ', ...' if len(ids) > 12 else '')
            lines.append(f'  ell={ell}: diagram(s) {shown}: {reason[:300]}')
    return '\n'.join(lines)


def _real_inputs(num_params, diagram_records):
    """Real parameters and diagram prefactors: with a real propagator the
    Phase J result at conj(w) is conj of the one at w."""
    from sage.all import CDF
    try:
        for v in (num_params or {}).values():
            if complex(CDF(SR(v))).imag != 0:
                return False
        for r in diagram_records:
            pref = SR(r['combined_prefactor']).subs(num_params or {})
            if complex(CDF(pref)).imag != 0:
                return False
    except Exception:
        return False
    return True


def _combine_phase_j(results, nodes):
    """One ell's Phase J results of the deformation nodes, combined: the
    callables, the per-diagram contributions and the delta records are the
    node-weighted sums; ``pole_deformation_nodes`` lists the nodes."""
    from api import _pole_clusters as _pc
    if all(r is None for r in results):
        return None
    out = dict(results[0])
    out['total_C'] = _pc.combine_callables(
        [r['total_C'] for r in results], nodes)
    fns = [r.get('total_C_batch') for r in results]
    if all(f is not None for f in fns):
        def total_C_batch(tau_points, *a, **kw):
            vals = [np.asarray(f(tau_points, *a, **kw), dtype=complex)
                    for f in fns]
            return list(_pc.combine_values(vals, nodes))
        out['total_C_batch'] = total_C_batch
    epd = [r.get('eval_per_diagram_batch') for r in results]
    if all(f is not None for f in epd):
        def eval_per_diagram_batch(tau_points, *a, **kw):
            outs = [f(tau_points, *a, **kw) for f in epd]
            vals = [np.asarray(o[0], dtype=complex) for o in outs]
            return _pc.combine_values(vals, nodes).tolist(), outs[0][1]
        out['eval_per_diagram_batch'] = eval_per_diagram_batch
    groups = []
    for gs in zip(*[r.get('groups') or [] for r in results]):
        g = dict(gs[0])
        if all(x.get('contribution') is not None for x in gs):
            g['contribution'] = _pc.combine_callables(
                [x['contribution'] for x in gs], nodes)
        groups.append(g)
    out['groups'] = groups
    # A diagram that failed at ANY node is missing from that node's total:
    # report the union (the per-node records are in node order).
    sk_all, seen = [], set()
    for r in results:
        for sk in (r or {}).get('skipped_kernel_ids') or []:
            # by diagram / group id: the reason text may differ by node
            key = ((sk.get('kernel_id'), sk.get('group_idx'))
                   if isinstance(sk, dict) else sk)
            if key not in seen:
                seen.add(key)
                sk_all.append(sk)
    out['skipped_kernel_ids'] = sk_all
    for g, gs in zip(groups, zip(*[r.get('groups') or [] for r in results])):
        bad = [x for x in gs if x.get('handled_by') == 'skipped']
        if bad and g.get('handled_by') != 'skipped':
            g['handled_by'] = 'skipped'
            g['reason'] = bad[0].get('reason')
            g['contribution'] = None
    for key in ('delta_contributions', 'delta_contributions_multi'):
        if key not in out:
            continue
        dcs = []
        for r, (w, wt, paired) in zip(results, nodes):
            for rec in r.get(key) or []:
                rec = dict(rec)
                f = rec.get('coeff_fc')
                if callable(f):
                    rec['coeff_fc'] = _pc.combine_callables([f],
                                                            [(w, wt, paired)])
                rec['pole_deformation_node'] = w
                dcs.append(rec)
        out[key] = dcs
    out['pole_deformation_nodes'] = [dict(w=w, weight=wt, paired=paired)
                                     for w, wt, paired in nodes]
    return out


def _phase_j_memory_preflight(diagram_records, propagator_data, num_params,
                              prop, model, k, max_ell, verbose, preflight,
                              ss_route=None, n_tau=1):
    """Item 35: the estimate of the modesum-plan memory of the whole Phase J
    run (``final_integral.estimate_modesum_plans`` over every diagram of
    every ell, times the pole-deformation node runs), recorded in the
    analyzer report (``diagnostics['phase_j_memory']`` and an assertion in
    the ledger) and returned for ``result['phase_j_memory']``.  The estimate
    is an upper bound (δ-subsets that build no plan are counted).  More
    than ``PHASE_J_PREPASS_REFUSE_FACTOR`` times over the budget
    ``PHASE_J_MEM_BUDGET_GB`` it raises ``PhaseJResourceError`` before any
    plan is built, under every policy; between 1x and that factor the
    running guard decides at each allocation.

    S3 (``ss_route``: the run's route prediction, ``_ss_route_prediction``):
    the diagrams the state-space route takes are charged its estimate
    (``statespace.diagram_estimate``: stationary states x the τ batch, and
    the largest segment working set) instead of pole tuples, and a run the
    route covers makes one Phase J pass (no deformation runs)."""
    from engine.integration.time_domain import final_integral as _fi
    if not hasattr(_fi, 'estimate_modesum_plans'):
        return None             # an A/B harness running an older engine
    deform = prop.get('pole_deformation')
    if ss_route is not None and ss_route.get('bypass_deformation'):
        deform = None
    ss_est = None
    if ss_route is not None and ss_route.get('covers_run'):
        from engine.integration.time_domain import statespace as _ssm
        n_perm = math.factorial(max(1, int(k)))

        def ss_est(td, has_delta):
            return _ssm.diagram_estimate(td, propagator_data, num_params,
                                         has_delta, n_tau=n_tau,
                                         n_perm=n_perm)
    # the node runs ``_phase_j_deformed`` makes: at most ``n_nodes`` (fewer
    # for a real system, which costs a scan of every prefactor to decide;
    # done only when the bound is over budget)
    runs = int(getattr(deform, 'n_nodes', 1) or 1) if deform is not None \
        else 1
    budget = _fi.phase_j_mem_budget_bytes()
    est = _fi.estimate_modesum_plans(
        [r['typed_diagram'] for r in diagram_records], propagator_data,
        num_params, prefactors=[r['combined_prefactor']
                                for r in diagram_records],
        budget_bytes=None if budget is None else budget / runs,
        **({} if ss_est is None else {'ss_estimate': ss_est}))
    if est is None:
        return None
    if (deform is not None and budget is not None
            and est['bytes'] * runs > budget):
        try:
            runs = len(deform.nodes(deform.real_system and _real_inputs(
                num_params, diagram_records)))
        except Exception:
            pass
    w = est['worst']
    worst_rec = diagram_records[w['index']] if w else None
    est.update(node_runs=runs, bytes=est['bytes'] * runs,
               tuples=est['tuples'] * runs,
               tuples_unpruned=est['tuples_unpruned'] * runs,
               budget_bytes=budget,
               worst_ell=worst_rec['ell'] if worst_rec else None,
               worst_index_in_ell=(
                   sum(1 for r in diagram_records[:w['index']]
                       if r['ell'] == worst_rec['ell'])
                   if worst_rec else None))
    factor = _fi.PHASE_J_PREPASS_REFUSE_FACTOR
    above = budget is not None and est['bytes'] > budget
    over = budget is not None and est['bytes'] > factor * budget
    est['above_budget'] = above
    detail = (f"modesum plans: <= {est['tuples']:,} pole tuples "
              f"(~{est['bytes'] / 1e9:.2f} GB; "
              f"{est['tuples_unpruned']:,} without the exact-zero pruning) "
              f"in {est['n_diagrams']} diagrams"
              + (f" x {runs} deformation runs" if runs > 1 else '')
              + (f"; state-space route: {est['ss_diagrams']} diagrams, "
                 f"<= {est['ss_bytes'] / 1e9:.3g} GB of stationary states "
                 f"+ {est['ss_transient_bytes'] / 1e9:.3g} GB of segment "
                 f"working set" if est.get('ss_diagrams') else '')
              + (f"; budget {budget / 1e9:.3g} GB (PHASE_J_MEM_BUDGET_GB)"
                 if budget is not None else '; no budget')
              + ('; above the budget: the plans are checked as they are '
                 'built' if above and not over else ''))
    if preflight is not None:
        preflight.diagnostics['phase_j_memory'] = dict(est)
        preflight.ctx.ledger_add('assertion', 'phase_j_memory_budget',
                                 detail, satisfied=not over)
    if verbose:
        print(f'  [memory] {detail}')
    if over:
        if not w or not w['edges']:
            raise _fi.PhaseJResourceError(
                f"Phase J memory estimate more than {factor}x over budget "
                f"for {model.get('name', '<unnamed>')} at k={k}, "
                f"max_ell={max_ell}: {detail}.  Nothing was built.  "
                f"Options: {_fi._BUDGET_HOW}.", est)
        edges = ', '.join(
            f"{n}{'' if n == a else f'/{a}'}{'+delta' if dl else ''}"
            for (n, a, dl) in w['edges'])
        raise _fi.PhaseJResourceError(
            f"Phase J memory estimate more than {factor}x over budget for "
            f"{model.get('name', '<unnamed>')} at k={k}, max_ell={max_ell}: "
            f"{detail}.  Largest diagram: loop order {est['worst_ell']}, "
            f"#{est['worst_index_in_ell']} of that order, {w['tuples']:,} "
            f"tuples ({w['tuples_unpruned']:,} unpruned) over "
            f"{len(w['edges'])} propagator edges with [{edges}] modes "
            f"(nonzero/all).  Nothing was built.  Options: "
            f"{_fi._BUDGET_HOW}.", est)
    return est


def _phase_j_ss_bypass(phase_j_pass, propagator_data, engine_report):
    """S3: one Phase J pass without the pole-cluster deformation, for a run
    the state-space route is predicted to cover.  Every subset must then
    take the route: a subset that declines it (its rows) or falls back on
    the pole route (which would use the undeformed poles and residues)
    makes this return ``None`` (the caller runs the deformation instead;
    the memory charged by this pass is taken back).  The pass's propagator
    data carry ``ss_no_pole_fallback``, so a fallback raises: in this pass
    (caught here) and when the returned callables are evaluated later."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import statespace as _ssm
    C = _ssm._counters()
    before = (C.get('ss_subsets_declined', 0), C.get('ss_fallback_evals', 0))
    # a failed bypass leaves no trace in the run's counters (its pass is
    # discarded): restored below
    snap = {k_: (dict(v) if isinstance(v, dict) else v)
            for k_, v in C.items()}
    with _fi._PLAN_MEM_LOCK:
        mem0 = {k_: _fi._PLAN_MEM[k_] for k_ in ('used', 'tuples', 'plans')}
    why = None
    try:
        out = phase_j_pass(dict(
            propagator_data, ss_no_pole_fallback=(
                'the run skipped the pole-cluster deformation (the '
                'state-space route covers it), so the undeformed poles of '
                'this propagator cannot serve a pole-route fallback; set '
                'DAEDALUS_PHASE_J_ENGINE=poles for this model')))
    except (_ssm.SSNoPoleFallback, _ssm.SSLateRefusal) as exc:
        # a fallback refused in the pass (SSLateRefusal: with
        # DAEDALUS_PHASE_J_SS_FALLBACK=0); anything else propagates
        out, why = None, f'{type(exc).__name__}: {str(exc)[:200]}'
    after = (C.get('ss_subsets_declined', 0), C.get('ss_fallback_evals', 0))
    if out is not None and after != before:
        out, why = None, (f'{after[0] - before[0]} subset(s) declined the '
                          f'state-space route, {after[1] - before[1]} '
                          f'fallback evaluation(s)')
    if out is not None:
        # δ records of m = 0 subsets with smooth rows take their factors from
        # the pole-residue G(t) (not routed): with undeformed poles they
        # would be the clustered ones
        for res_ell in (out[0] or {}).values():
            for key in ('delta_contributions', 'delta_contributions_multi'):
                for rec in (res_ell or {}).get(key) or ():
                    if not rec.get('integrated') and rec.get(
                            'retardation_data'):
                        out, why = None, ('a delta record with smooth rows '
                                          'outside the state-space route')
                        break
                if out is None:
                    break
            if out is None:
                break
    if out is None:
        with _fi._PLAN_MEM_LOCK:
            if _fi._PLAN_MEM['active'] > 0:
                _fi._PLAN_MEM.update(mem0)
        for k_ in list(C):
            v = snap.get(k_)
            if isinstance(C[k_], dict):      # in place: callers hold them
                C[k_].clear()
                C[k_].update(v or {})
            else:
                C[k_] = 0 if v is None else v
        engine_report['pole_deformation'] = 'used'
        engine_report['bypass_failed'] = why
    return out


def _phase_j_deformed(phase_j_pass, propagator_data, deform, diagram_records,
                      num_params, tau_points, report):
    """Phase J for a propagator with resolved pole clusters: one pass per
    deformation node (``api._pole_clusters.PoleDeformation``), combined by
    the trapezoid rule on the unit circle; ``report`` gets the node count
    and the accuracy estimate (discrete Fourier tail, relative to the
    largest |C| on the grid), and a warning is issued above
    ``ACCURACY_WARN``."""
    from api import _pole_clusters as _pc
    symmetric = deform.real_system and _real_inputs(num_params,
                                                    diagram_records)
    nodes = deform.nodes(symmetric)
    runs = []
    for w, _, _ in nodes:
        pv, cm = deform.propagator_at(w)
        runs.append(phase_j_pass(dict(propagator_data, pole_vals=pv,
                                      C_mats=cm)))
    phase_j_by_ell, total_C_by_ell, C_tau_by_ell = {}, {}, {}
    acc_ell, err_tot, node_tot = {}, None, None
    for ell in runs[0][1]:
        total_C_by_ell[ell] = _pc.combine_callables(
            [r[1][ell] for r in runs], nodes)
        phase_j_by_ell[ell] = _combine_phase_j([r[0][ell] for r in runs],
                                               nodes)
        if tau_points is None:
            C_tau_by_ell[ell] = None
            continue
        vals = [np.asarray(r[2][ell], dtype=complex) for r in runs]
        C_tau_by_ell[ell] = _pc.combine_values(vals, nodes)
        err = _pc.tail_estimate(vals, nodes, deform.n_nodes)
        scale = float(np.max(np.abs(C_tau_by_ell[ell]))) if vals[0].size else 0
        acc_ell[ell] = (float(np.max(err)) / scale if scale > 0
                        else float(np.max(err)) if err is not None else None)
        node_tot = vals if node_tot is None else [a + b for a, b in
                                                  zip(node_tot, vals)]
    report.update(n_nodes=deform.n_nodes, node_runs=len(nodes),
                  symmetric=bool(symmetric))
    if tau_points is None or node_tot is None:
        report.update(accuracy=deform.a_priori_error, accuracy_source='a priori',
                      accuracy_by_ell=None, accuracy_by_tau=None,
                      accuracy_l1=deform.a_priori_error)
    else:
        err = _pc.tail_estimate(node_tot, nodes, deform.n_nodes)
        C = _pc.combine_values(node_tot, nodes)
        absC = np.abs(C)
        scale = float(np.max(absC)) if absC.size else 0.0
        acc = float(np.max(err)) / scale if scale > 0 else float(np.max(err))
        with np.errstate(divide='ignore', invalid='ignore'):
            by_tau = np.where(absC > 0, err / np.where(absC > 0, absC, 1),
                              np.where(err > 0, np.inf, 0.0))
        l1 = float(np.sum(err)) / float(np.sum(absC)) if np.sum(absC) > 0 \
            else float(np.sum(err))
        report.update(accuracy=acc, accuracy_source='fourier tail',
                      accuracy_by_ell=acc_ell,
                      accuracy_by_tau=[float(x) for x in by_tau],
                      accuracy_l1=l1)
    if report['accuracy'] is not None and report['accuracy'] > _pc.ACCURACY_WARN:
        _pc.warn(f'pole-cluster deformation: estimated relative error '
                 f'{report["accuracy"]:.3g} (above {_pc.ACCURACY_WARN:g}) '
                 f'with {deform.n_nodes} nodes; see result["pole_clusters"].')
    return phase_j_by_ell, total_C_by_ell, C_tau_by_ell


def _trunc(s, maxlen=200):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen - 3] + '...'


def _print_action_sectors(ft):
    """Print MF / Free / Interaction action sectors after expand()."""
    ring_gen_names = [str(g) for g in ft.ring().gens()]

    def _fmt_poly(poly):
        lines = []
        for exp_vec, coeff in poly.dict().items():
            mono = '·'.join(
                f'{ring_gen_names[i]}^{exp_vec[i]}' if exp_vec[i] > 1
                else ring_gen_names[i]
                for i in range(len(exp_vec)) if exp_vec[i] > 0
            )
            mono = mono if mono else '1'
            lines.append(f'        {mono}  *  ({_trunc(coeff, 220)})')
        return lines

    print()
    print('      ── MF action (bigrade ≤1 in each index; vanishes at saddle) ──')
    mf_raw = getattr(ft, '_mf_sector_raw', None) or {}
    any_mf = False
    for key in [(0, 0), (1, 0), (0, 1)]:
        poly = mf_raw.get(key)
        if poly is None or not poly.dict():
            continue
        any_mf = True
        print(f'      bigrade {key}:')
        for line in _fmt_poly(poly):
            print(line)
    if not any_mf:
        print('        (empty)')

    print()
    print('      ── Free action (1,1) bilinear sector ──')
    S_free = ft.free_action()
    if not S_free.dict():
        print('        (empty)')
    else:
        for line in _fmt_poly(S_free):
            print(line)

    print()
    print('      ── Interaction action (total degree ≥ 2, excluding (1,1)) ──')
    sectors = ft.sectors()
    any_int = False
    for (n_t, n_p), poly in sorted(sectors.items()):
        if (n_t, n_p) == (1, 1) or n_t + n_p < 2:
            continue
        if not poly.dict():
            continue
        any_int = True
        print(f'      bigrade ({n_t},{n_p}):')
        for line in _fmt_poly(poly):
            print(line)
    if not any_int:
        print('        (empty)')
    print()


class _PipelineState:
    """The pipeline objects after the mean-field stage, handed to the
    chosen backend's ``run`` (``api.routing``).  Plain attributes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _preflight_default() -> bool:
    """``DAEDALUS_PREFLIGHT`` (read at call time): ``0``/``off``/``false``
    turns the analyzer gates off; default on."""
    v = os.environ.get('DAEDALUS_PREFLIGHT', '').strip().lower()
    return v not in ('0', 'off', 'false', 'no')


def _make_preflight(preflight, model, k, max_ell, external_fields_user,
                    fundamental, fixed_point_index, policy, accuracy,
                    taylor_order):
    """The analyzer for this call, or ``None`` when it is off."""
    if preflight is None:
        preflight = _preflight_default()
    if not preflight:
        return None
    from api.analyze import Preflight, Question
    q = Question(k=k, max_ell=max_ell,
                 external_fields=tuple(tuple(e) for e in external_fields_user),
                 parameters=dict(fundamental) if fundamental else None,
                 fixed_point_index=fixed_point_index, accuracy=accuracy)
    return Preflight(model, q, policy, taylor_order=taylor_order)


#: Phase J counters that are maxima (G6 resets them around a run and
#: restores ``max(before, run)``, so a caller that reads them after the
#: call sees the same value as without the analyzer).
_MAX_COUNTERS = ('dd_err_ratio_max', 'dbm_error_ratio_max',
                 'nquad_hardened_cap_span_max')


def _counter_snapshot():
    """The Phase J counters and worker-state identities before a run."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import pipeline as _pl
    c = _fi._RUNTIME_COUNTERS
    snap = {k_: v for k_, v in c.items() if not isinstance(v, dict)}
    # S3: the state-space route's counters by reason (dicts)
    for k_ in _SS_DICT_COUNTERS:
        if isinstance(c.get(k_), dict):
            snap[k_] = dict(c[k_])
    maxes = {k_: c[k_] for k_ in _MAX_COUNTERS if k_ in c}
    for k_ in maxes:
        c[k_] = 0.0
    workers = {k_: id(v) for k_, v in _pl._WORKER_STATE.items()}
    return snap, maxes, workers


_SS_DICT_COUNTERS = ('ss_declined_by_reason', 'ss_fallback_by_reason')


def _counter_delta(token):
    """``(counters of the run, complete)``: additive counters as
    differences, maxima as the run's maximum (the global value restored to
    ``max(before, run)``); ``complete`` is False when ``total_C_batch``
    handed work to forked workers (their counters stay in the children)."""
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import pipeline as _pl
    snap, maxes, workers = token
    c = _fi._RUNTIME_COUNTERS
    out = {}
    for k_, v in c.items():
        if isinstance(v, dict):
            if k_ in _SS_DICT_COUNTERS:
                b = snap.get(k_) if isinstance(snap.get(k_), dict) else {}
                d = {r: n - b.get(r, 0) for r, n in v.items()
                     if n - b.get(r, 0)}
                out[k_] = d
            continue
        if k_ in maxes:
            out[k_] = v
            c[k_] = max(maxes[k_], v)
        else:
            out[k_] = v - snap.get(k_, 0)
    now = {k_: id(v) for k_, v in _pl._WORKER_STATE.items()}
    complete = now == workers
    return out, complete


class PhaseJEngineFallbackWarning(UserWarning):
    """``compute_cumulants``: the state-space route fell back on the pole
    route at call time for some subsets (``result['phase_j_engine']``
    counts them by reason; the values are the pole route's there)."""


def _engine_census(result, counters):
    """``result['phase_j_engine']`` (S3): the run's prediction (set by the
    Phase J backend) plus the census of the counters: subsets built on the
    state-space route, declined by reason, fallbacks by reason, and
    ``statespace_share`` (routed / all subsets evaluated).  A fallback warns
    once (``PhaseJEngineFallbackWarning``: never silent).  When subsets
    declined the route although the run was predicted covered, a pole
    cluster report marked 'bypassed' gets its pole status back (those
    subsets used the poles)."""
    rep = result.get('phase_j_engine')
    if rep is None:
        return
    c = counters or {}
    n_ss = int(c.get('ss_subsets_built', 0))
    n_dec = int(c.get('ss_subsets_declined', 0))
    rep.update(
        ss_subsets=n_ss, ss_declined=n_dec,
        ss_declined_by_reason=dict(c.get('ss_declined_by_reason') or {}),
        ss_evals=int(c.get('ss_evals', 0)),
        fallback_subsets=int(c.get('ss_fallback_subsets', 0)),
        fallback_evals=int(c.get('ss_fallback_evals', 0)),
        fallback_by_reason=dict(c.get('ss_fallback_by_reason') or {}),
        statespace_share=(n_ss / (n_ss + n_dec)) if n_ss + n_dec else None)
    pc = result.get('pole_clusters')
    if (isinstance(pc, dict) and pc.get('status') == 'bypassed'
            and rep.get('pole_deformation') != 'bypassed'
            and (n_dec or rep['fallback_evals'])):
        pc['status'] = pc.get('status_poles', pc['status'])
        pc['reason'] = pc.get('reason_poles', pc.get('reason'))
    if rep['fallback_evals']:
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(PhaseJEngineFallbackWarning, 'always')
        warnings.warn(
            f'Phase J: the state-space route fell back on the pole route for '
            f'{rep["fallback_subsets"]} subset(s) ({rep["fallback_evals"]} '
            f'evaluation(s); reasons {rep["fallback_by_reason"]}); see '
            f'result["phase_j_engine"]', PhaseJEngineFallbackWarning,
            stacklevel=3)


def _attach_provenance(result, route, pf):
    """The route and the analyzer report on the result (``result['route']``,
    ``result['analysis']``; ``config['route']`` / ``config['assumptions']``
    for ``api.save``)."""
    rep = pf.report() if pf is not None else None
    result['route'] = route.as_dict()
    result['analysis'] = rep
    cfg = result.get('config')
    if isinstance(cfg, dict):
        cfg['route'] = route.name
        cfg['assumptions'] = rep.ledger_dicts() if rep is not None else None



# ── switch-on transients (item 44) ─────────────────────────────────────────

def _switch_on_request(model, k, max_ell, t_min, times, initial_covariance,
                       tau_grid, *, spatial, grouped, saving):
    """Validate the switch-on arguments of ``compute_cumulants`` (before any
    work).  ``None`` for a stationary run (``t_min=None``); otherwise the
    request ``{'t_min', 'times' (n, k), 'initial_covariance'}``."""
    extra = [n for n, v in (('times', times),
                            ('initial_covariance', initial_covariance))
             if v is not None]
    if t_min is None:
        if extra:
            raise ValueError(
                f'{", ".join(extra)} need t_min (a switch-on transient); '
                f'without t_min the run is stationary')
        return None
    try:
        t0 = float(t_min)
    except (TypeError, ValueError):
        raise ValueError(f't_min must be a finite float, got {t_min!r}') \
            from None
    if not math.isfinite(t0):
        raise ValueError(f't_min must be finite (omit it for the stationary '
                         f'state), got {t_min!r}')
    from engine.integration.time_domain import statespace as _ss
    if spatial:
        raise NotImplementedError(
            'switch-on transients (t_min) are implemented for temporal '
            'models only')
    if grouped:
        raise ValueError(
            't_min: the grouped integrator (use_grouped_phase_j=True) is not '
            'supported; use the per-diagram path (the default)')
    if saving:
        raise ValueError(
            't_min: output_npz / output_csv do not support switch-on results '
            'yet; save result["times"] and result["C_times"] instead')
    if tau_grid is not None:
        raise ValueError(
            't_min: tau_grid is the stationary lag grid; a switch-on run is '
            'evaluated at the absolute times given by times=')
    if not _ss.engine_requested():
        raise _ss.SwitchOnUnsupported(
            'engine_poles', f'{_ss.ENGINE_ENV}=poles selects the pole engine')
    if times is None:
        raise ValueError('t_min needs times=: the absolute external times, '
                         f'shape (n, k={k}) or (n,)')
    T = np.asarray(times, dtype=float)
    if T.ndim == 0:
        T = T.reshape(1)
    if T.ndim == 1:
        T = np.repeat(T[:, None], k, axis=1)
    if T.ndim != 2 or T.shape[1] != k or T.shape[0] == 0:
        raise ValueError(f'times must have shape (n, k={k}) or (n,), got '
                         f'{np.shape(times)}')
    if not np.all(np.isfinite(T)):
        raise ValueError('times must be finite')
    if np.any(T < t0):
        raise ValueError(
            f'every time must be >= t_min = {t0!r} (the fluctuations start '
            f'at t_min); the earliest is {float(T.min())!r}')
    C0 = None
    if initial_covariance is not None:
        if k != 2:
            raise ValueError(f'initial_covariance applies to k = 2 only (got '
                             f'k = {k})')
        if max_ell != 0:
            raise NotImplementedError(
                f'initial_covariance with loops (max_ell = {max_ell}) is not '
                f'implemented: the initial-state term is the tree-level one '
                f'(loops need an initial-state vertex); pass max_ell=0')
        C0 = np.asarray(initial_covariance, dtype=float)
        if (C0.ndim != 2 or C0.shape[0] != C0.shape[1]
                or not np.all(np.isfinite(C0))):
            raise ValueError('initial_covariance must be a finite square '
                             'matrix')
    return {'t_min': t0, 'times': T, 'initial_covariance': C0}


def _switch_on_wrap_td(td, t_min):
    """Wrap the callables a switch-on run stores in ``phase_j_by_ell[ell]``
    (``total_C``, ``total_C_batch``, each group's ``contribution``) so that
    they, too, evaluate inside the scope (bare, they would give stationary
    values)."""
    if not isinstance(td, dict):
        return
    if callable(td.get('total_C')):
        fn = _switch_on_callable(td['total_C'], t_min)
        td['total_C'] = fn
        if 'total_C_batch' in td:
            td['total_C_batch'] = (
                lambda tau_points, *a, _fn=fn, **kw:
                [complex(_fn(*p)) for p in tau_points])
    for g in td.get('groups') or ():
        if isinstance(g, dict) and callable(g.get('contribution')):
            g['contribution'] = _switch_on_callable(g['contribution'], t_min)
    if callable(td.get('eval_per_diagram_batch')):
        # serial (no fork) and inside the scope
        _pd = td['eval_per_diagram_batch']
        _pd_scoped = _switch_on_callable(
            lambda tau_points: _pd(tau_points, parallel=False), t_min)
        td['eval_per_diagram_batch'] = (
            lambda tau_points, *a, _f=_pd_scoped, **kw: _f(tau_points))


def _switch_on_callable(fn, t_min):
    """A per-ell (or total) callable of absolute external times, evaluated
    inside ``statespace.switch_on_scope(t_min)`` (no Itô nudge: the
    state-space route applies the tie rule to equal times exactly)."""
    from engine.integration.time_domain import statespace as _ss

    def at(*ext_time_values):
        with _ss.switch_on_scope(t_min):
            return fn(*ext_time_values)
    return at


def _initial_state_term(so, propagator_data, num_params, external_fields,
                        phys_idx):
    """``(values at so['times'], callable)`` of the tree-level initial-state
    term ``R(t1-t_min) C0 R(t2-t_min)^T`` (``initial_covariance``), or
    ``None`` without one.  ``R``: the response from the model's realization
    (``statespace.realization_for``); it must be the identity at ``0+`` (a
    first-order model whose response index ``r`` is the equation of
    physical field ``r``), so that ``C0`` is the covariance of the
    fluctuations themselves.  ``C0`` enters like a noise vertex at
    ``t_min``, so at k = 2 this is the whole tree term."""
    C0 = so['initial_covariance']
    if C0 is None:
        return None
    from engine.integration.time_domain import statespace as _ss
    ops, why = _ss.realization_for(propagator_data, num_params)
    if ops is None:
        raise _ss.SwitchOnUnsupported(
            why, 'the initial-state term needs the model realization')
    nf = int(propagator_data['nf'])
    if C0.shape != (nf, nf):
        raise ValueError(
            f'initial_covariance must have shape {(nf, nf)} (the {nf} physical '
            f'field(s) in their internal order), got {C0.shape}')

    def R(p, r, t):
        ent = ops.entry(p, r)
        if ent is None or ent.n == 0:
            return 0.0
        v = ent.impulse(t)
        return float(np.real(v)) if abs(np.imag(v)) <= 1e-14 * (
            1 + abs(v)) else complex(v)

    for p in range(nf):
        for r in range(nf):
            v = R(p, r, 0.0)
            if abs(v - (1.0 if p == r else 0.0)) > 1e-12:
                raise ValueError(
                    'initial_covariance needs a first-order '
                    'model whose response at 0+ is the identity (response '
                    f'index r = the equation of physical field r); here '
                    f'R[{p}, {r}](0+) = {v!r}')
    legs = [phys_idx[f] for f in external_fields]
    t0 = so['t_min']

    def term(*ext):
        a = np.array([R(legs[0], r, ext[0] - t0) for r in range(nf)])
        b = np.array([R(legs[1], r, ext[1] - t0) for r in range(nf)])
        return complex(a @ C0 @ b)

    vals = np.array([term(*row) for row in so['times']], dtype=complex)
    return vals, term


def _embedded_noise_states(model):
    """Names of the auxiliary noise fields the model builder made from
    declared Markov-embeddable noise kernels (``correlated_noises`` entries
    ``*_markov_aux``).  Under a switch-on they start at zero like every
    fluctuation: the noise is switched on with the system."""
    cn = model.get('correlated_noises') or {} if isinstance(model, dict) \
        else {}
    return sorted(str(k) for k in cn if str(k).endswith('_markov_aux'))


def _ledger_switch_on(preflight, so):
    """The analyzer ledger records the non-stationary question: the
    stationary-state line is replaced by the switch-on one (the gates
    themselves are unchanged)."""
    if preflight is None:
        return
    ctx = preflight.ctx
    ctx.ledger[:] = [e for e in ctx.ledger if e.name != 'stationary_state']
    init = ('the fluctuations at t_min are zero'
            if so['initial_covariance'] is None else
            'plus the tree-level term of initial_covariance')
    emb = _embedded_noise_states(preflight.ctx.model)
    ctx.ledger_add('approximation', 'switch_on_state',
                   f'switch-on transient: fluctuations about the selected '
                   f'saddle start at t_min = {so["t_min"]!r} ({init}); every '
                   f'vertex time is > t_min; state-space Phase J route only'
                   + (f'; the Markov-embedded noise state(s) {emb} start at '
                      f'zero too (the noise is switched on at t_min)'
                      if emb else ''),
                   declared=True)


def _switch_on_check(result, failures, model_name):
    """A switch-on run is never partial: an unevaluated diagram, or δ
    components (whose weights are not yet derived for a transient), raise."""
    from engine.integration.time_domain import statespace as _ss
    if failures:
        raise PhaseJIncompleteError(
            'switch-on run (t_min): ' + _phase_j_failure_message(
                failures, model_name), result=result, failures=failures)
    for ell, td in (result.get('phase_j_by_ell') or {}).items():
        if td and (td.get('delta_contributions')
                   or td.get('delta_contributions_multi')
                   or td.get('n_delta_subsets_dropped')):
            raise _ss.SwitchOnUnsupported(
                'delta_components', f'ell = {ell} has δ(t) components (an '
                f'instantaneous response; recorded or dropped shot-noise '
                f'subsets); their switch-on weights are not implemented')


def compute_cumulants(
    model: dict,
    k: int,
    max_ell: int = 0,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    *,
    parameters: dict = None,         # canonical name for ``fundamental`` (wins if both given)
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    tau_grid=None,                   # explicit τ grid (array); overrides tau_max/tau_step
    spatial_grid=None,
    chi_grid=None,                   # canonical name for ``spatial_grid`` (wins if both given)
    taylor_order: int = None,
    origin_leaf_idx: int = 0,
    output_npz: str = None,
    output_csv: str = None,
    use_cache: bool = True,
    parallel: bool = True,
    spatial_parallel: bool = True,   # spatial THREADS (no fork — safe); smart-gated to L≥2 where it helps (~2.5×)
    spatial_n_q: int = 64,           # spatial loop q-grid count — cost is LINEAR in it (64→16 = 4× faster; coarsens C(x) at large x)
    spatial_points=None,             # k>=3 spatial: (n_pts, k-1, 2) of (x_j, tau_j) offsets per non-anchor slot

    n_workers: int = None,
    use_grouped_phase_j: bool = False,
    fixed_point_index: int = 0,
    mf_dae_n_starts: int = 64,
    mf_dae_seed_box: dict | None = None,
    verbose: bool = True,
    policy=None,                     # api.analyze.Policy (or dict): strict / consents / downgrade
    preflight: bool = None,          # the analyzer gates (default on; DAEDALUS_PREFLIGHT=0 turns them off)
    backend: str = None,             # an explicit backend name (api.routing); None = the router picks
    accuracy: float = None,          # target relative accuracy (G6 threshold; default 1e-10)
    t_min: float = None,             # switch-on transients: fluctuations start at 0 at this absolute time
    times=None,                      # with t_min: absolute external times, (n, k) or (n,) (every leg at t)
    initial_covariance=None,         # with t_min (k=2, tree): covariance of the fluctuations at t_min
    delay: dict = None,              # delay kernels (api.delay): order / tol / n_start / n_max / wall_max / tree
) -> dict[str, Any]:
    """
    Compute the k-point cumulant slice for ``model`` at the given
    parameter point, all the way through Phase J time-domain integration.

    Parameters
    ----------
    model : dict
        The model dict (e.g. ``HAWKES_MODEL`` from
        ``simulations/hawkes_linear_expg_gtas.py``).  Must declare:
        ``response_fields``, ``physical_fields``, ``parameters``,
        ``kernels``, ``operators``, ``functions``, ``mf_substitutions``,
        ``mf_bg_conditions``, ``specializations``, ``kernel_ft_image``,
        ``phi_concrete``, ``mf_equations``, ``action``.
        Optional: ``correlated_noises``.
    k : int
        Number of external legs (k-point cumulant).
    max_ell : int, default 0
        Maximum loop order (0 = tree only).
    fundamental : dict
        Numerical parameter values keyed by param name.  E.g.::

            {'a': 1.0, 'tau': 10.0, 'tau_g': 2.5,
             'E': [1.1, 1.05], 'w': [[0.35, 0.4], [0.3, 0.5]],
             'w_X': 0.3, 'lambda_X': 2.5, 'p_part': 0.6,
             'mu_shift_diff': 0.0, 'sigma_shift_diff_sq': 1.0}

    external_fields : list of (str, int)
        Length-k list of leaf field tuples, e.g. ``[('dn', 1), ('dn', 2)]``.
    tau_max, tau_step : float
        τ-grid extent and spacing for the Phase J slice evaluation.
    taylor_order : int or None
        Truncation order for FieldTheory's nonlinear-function Taylor.
        ``None`` (default) auto-picks ``max(k + 2 * max_ell, 2)``,
        which is the smallest order that captures every vertex needed
        for a connected diagram with ``k`` external legs at ``ell``
        loops (max vertex order ≤ ``k + 2·ell``).  The floor of 2 is
        the structural minimum — order 2 still has the (1,1) bilinear
        propagator kernel plus the (0,0)/(1,0)/(0,1) MF saddle sectors,
        which is everything the cache machinery needs even for the
        degenerate ``k=2, max_ell=0`` case (tree-level pair correlator
        = bare propagator, no interaction vertices).  Pass an explicit
        integer to override (e.g. for non-standard diagrammatic
        content or when probing higher-order vertices for cache
        invalidation testing).

        **Previously this floor was 4**, which forced models at
        ``k=2, max_ell=0`` to pay an order-4 expansion they didn't
        mathematically need — see ``docs/CHANGELOG.md``
        (model-precompute-cache branch) for the cost it imposed on
        the dendritic-linear model.
    origin_leaf_idx : int
        Which canonical position to pin to t=0 for the slice (default 0).
    output_npz : str or None
        If given, save the result to this ``.npz`` path via
        ``api.save.save_npz``.  See that module for the schema —
        per-loop-order curves (``C_tree``, ``C_1_loop``, ...,
        ``C_total``), adaptive mean-field arrays, parameter keys.
    output_csv : str or None
        If given, save a wide-format CSV companion to ``output_npz``
        via ``api.save.save_csv`` (parameter / mean-field
        metadata as comment lines, then a τ-grid table).
    use_cache : bool
        Whether to reuse cached symbolic propagator (per ``(model, taylor)``)
        and unique typed diagrams (per ``(model, taylor, k, ell, ext_fields)``).
        Both caches live as sibling files under ``saved_models/<model>/``
        (e.g. ``saved_models/<model>/expand_taylor<N>.sobj``), one per
        Taylor order.  With ``False`` no cache file is read or written, and
        a temporal run takes its prediagrams from the eager enumerator, as
        before the streaming enumerator existed (``DAEDALUS_PREDIAGRAM_EAGER=0``
        selects the faster streaming one; see
        ``engine.enumeration.prediagram_cache.TEMPORAL_CACHE_OFF_STREAMS``).
        Cache on and cache off can use different labelled representatives of
        the same diagrams (in a different ``result['diagrams']`` order):
        totals then agree to rounding, or to the ``scipy.nquad`` fallback's
        tolerance on models whose regions fall back to it.  A spatial run
        never uses the prediagram cache; it streams its prediagrams in
        memory whatever this says.
    parallel : bool, default True
        Enable fork-based multiprocessing for the two heavy stages
        that support it: per-prediagram type assignment in step [5]
        and per-τ Phase J evaluation in step [7].  Both stages share
        the same flag.
    n_workers : int or None, default None
        Worker process count when ``parallel=True``.  ``None`` lets
        each stage pick its own default (``min(os.cpu_count(),
        n_tasks)``).
    verbose : bool
        Print progress messages.
    policy : api.analyze.Policy, dict or None
        What the model analyzer may accept without asking: ``mode``
        (``'permissive'`` default, ``'strict'`` makes warnings errors),
        consents (``allow_taylor_truncation``, ``allow_markov_embedding``,
        ``allow_numeric_fallback``), ``interpretation``, ``assertions`` and
        ``downgrade`` (error codes accepted as warnings).
    preflight : bool or None
        Run the analyzer gates (``api.analyze.Preflight``) at the stage
        boundaries: G0/G1 before the expansion, G1 after it, G3 (class)
        after the propagator, G2/G4 after the mean field, G3 (poles) after
        the residues, all before the diagram enumeration; any ``error``
        finding raises :class:`api.analyze.AnalyzerError` (a
        ``ValueError``) with its code and remedy.  ``None``: on, unless the
        environment sets ``DAEDALUS_PREFLIGHT=0``.  The gates read the
        pipeline's own objects and change no number.
    backend : str or None
        Force one registered backend (``api.routing.backends()``); it must
        still apply.  ``None``: the first applicable one.
    accuracy : float or None
        Target relative accuracy for the G6 total-relative accuracy check
        (``Question.accuracy``; default ``1e-10``, the per-subset acceptance
        of the Phase J routes).
    t_min : float or None
        ``None`` (default): the stationary state (time-translation
        invariant; nothing below applies).  A finite value asks for a
        **switch-on transient**: the fluctuations about the selected
        mean-field saddle start at zero at the absolute time ``t_min``
        (the state is the saddle exactly at ``t_min``), so every vertex-time
        integral has the lower bound ``t_min`` and the cumulant is a function
        of the absolute external times.  Computed on the state-space Phase J
        route only (``engine.integration.time_domain.statespace``: a finite
        first segment from the empty state); a model or subset it does not
        take, ``DAEDALUS_PHASE_J_ENGINE=poles``, spatial models, the grouped
        integrator and ``output_npz``/``output_csv`` raise.  As ``t_min ->
        -inf`` the result tends to the stationary one.  A noise kernel
        the model builder Markov-embedded (an auxiliary field) starts at
        zero too: the noise is switched on with the system
        (``result['nonstationary']['embedded_noise_states']``).
    times : array or None
        Required with ``t_min``: the absolute external times to evaluate,
        shape ``(n, k)`` (row ``i`` = the times of the ``k`` legs, in the
        order of ``external_fields``) or ``(n,)`` (every leg at ``t``: the
        equal-time cumulant); taken literally (a tuple too: the ``(lo, hi,
        n)`` range form is ``dd.Config``'s).  Every time must be ``>=
        t_min``; equal times
        are exact ties (no Itô nudge: the state-space route applies the
        tie rule).  Values: ``result['C_times']``.  ``tau_grid`` (the
        stationary lag grid) must not be given with ``t_min``.
    initial_covariance : array or None
        With ``t_min``, ``k = 2``, ``max_ell = 0``: an initial covariance
        ``C0`` of the fluctuations at ``t_min`` (square, over the physical
        fields in their internal order, ``result['propagator']['nf']``).
        The tree adds the deterministic term ``R(t1 - t_min) C0
        R(t2 - t_min)^T`` (``R``: the response), which is the whole tree
        term of a Gaussian initial state (``C0`` enters like a noise vertex
        at ``t_min``); loops with ``C0 != 0`` are not implemented (raises).
        Needs a first-order model whose response at ``0+`` is the identity.
        (A deterministic initial offset of the mean is not offered: its tree
        is the full nonlinear relaxation, a time-dependent saddle.)
    delay : dict or None
        Options for models with delay kernels (``exp(-i omega d)`` factors in
        a kernel image; see :mod:`api.delay`): ``order`` (``'auto'`` or a
        fixed Pade order ``n``), ``tol`` (target relative accuracy of the
        observable, default ``1e-8``), ``n_start`` / ``n_max`` (doubling search,
        default 4 / 16), ``wall_max`` (cap in s on the predicted wall time of
        the next order, default 900), ``tree`` (``'exact'``, default: the
        k = 2 tree from the exact delayed covariance; ``'pade'``).  Ignored
        for models without delays.  The result then carries
        ``result['delay']``.

    Returns
    -------
    dict with keys:
        'total_C'         : callable(*tau) — full sum across ells
        'total_C_by_ell'  : {ell: callable(*tau)} — per-loop-order
        'C_tau'           : ndarray of total_C on the τ-grid (k=1, 2;
                             None for k≥3 — caller uses total_C / per-
                             ell callables instead)
        'C_tau_by_ell'    : {ell: ndarray or None} — per-loop-order
                             grid eval; ``C_tau == sum(values())``
        'tau_grid'        : ndarray of τ values
        'mf_values'       : raw {internal_name: [v_pop1, ...]}, adaptive
        'mf'              : :class:`api.access.MeanField` accessor —
                             ``mf['v', 1]`` returns ``v*_1``,
                             ``mf['n']`` returns the whole nstar vector
        'params'          : :class:`api.access.Parameters` accessor —
                             ``params['E', 1]``, ``params['w', 1, 2]``,
                             ``params['tau']`` (1-based indexing)
        'num_params'      : {SR symbol: float}
        'propagator'      : the propagator data dict
        'diagrams'        : list of dicts {'typed_diagram', 'classify',
                             'combined_prefactor', 'multiplicity' (M(F),
                             prod n_leg!/|Aut_ext|), 'dedup_class_size',
                             'ell'}
        'kernel_groups'   : list of dicts as fed to compute_correction_td
        'phase_j_by_ell'  : {ell: td_result dict}
        'status'          : (Phase J backends) ``'ok'``, or ``'incomplete'``
                             when Phase J could
                             not evaluate some diagram(s): they are left out
                             of ``C_tau``, a :class:`PhaseJIncompleteWarning`
                             names them (``policy={'mode': 'strict'}``
                             raises :class:`PhaseJIncompleteError`)
        'phase_j_incomplete' : {ell: [{'diagram', 'n_diagrams', 'reason'}]}
                             of those diagrams (``{}`` when complete)
        'route'           : the routing record (``api.routing.Route.
                             as_dict``: backend, rejected backends)
        'analysis'        : the :class:`api.analyze.ModelReport` of the
                             gates (traits, findings, assumptions ledger,
                             G6 run diagnostics); ``None`` with
                             ``preflight=False``
        'config'          : input args echoed back; in addition to the
                             obvious keys, ``external_fields_in`` echoes
                             the user-passed form (e.g. ``[('n', 1)]``)
                             while ``external_fields`` stores the
                             internal form (e.g. ``[('dn', 1)]``), and
                             ``phase_j_convention`` records the Phase J
                             equal-time rule the numbers were computed
                             with (``api.save`` stamps saved files with
                             it), ``route`` the backend name and
                             ``assumptions`` the ledger (saved too).

    SWITCH-ON (``t_min`` given): ``C_tau`` / ``C_tau_by_ell`` / ``tau_grid``
    are ``None``; instead
        'times'           : ndarray ``(n, k)`` of absolute external times
        'C_times'         : ndarray ``(n,)`` of the cumulant at each row
        'C_times_by_ell'  : {ell: ndarray (n,)}; ``C_times == sum(values())``
        't_min'           : the switch-on time
        'nonstationary'   : the record of the question (protocol, t_min,
                             initial-state terms and their values)
    and ``total_C`` / ``total_C_by_ell`` take ABSOLUTE times (each call runs
    inside :func:`statespace.switch_on_scope`).

    SPATIAL k≥3 (``spatial_points`` given) returns a DIFFERENT dict (an
    early return), keyed by the explicit evaluation events rather than a
    τ-grid:
        'C_kpoint'        : ndarray ``(n_pts,)`` — the k-point cumulant
                             at each event configuration
        'C_kpoint_by_ell' : {ell: ndarray (n_pts,)} per-loop-order
        'points'          : the ``spatial_points`` array echoed back
                             ``(n_pts, k-1, 2)`` of ``(x_j, τ_j)`` offsets
        'spatial_info'    : driver diagnostics (μ, D, per-ell, eig, …)
        'k', 'max_ell', 'mf'
    """
    # the call as given (a model with delay kernels is handed to api.delay,
    # which re-enters compute_cumulants with Pade-approximated copies)
    _delay_call = dict(locals())
    # Accept the canonical names the dd.Config layer uses (``parameters`` /
    # ``chi_grid``), mirroring onto the internal names; canonical wins.
    if parameters is not None:
        fundamental = parameters
    if chi_grid is not None:
        spatial_grid = chi_grid
    # Fall back to the defaults declared via ``ModelBuilder.parameter(...,
    # default=...)`` for every parameter the caller did not give: an EMPTY
    # dict (several models ship ``DEFAULT_FUNDAMENTAL = {}``, and a caller
    # forwarding that verbatim must not sail past this and fail much later
    # with a bare ``NameError`` from inside the mean-field solve) and a
    # PARTIAL one alike -- the given values win.  Mean-field parameters
    # (``mean_field=True``, e.g. ``xstar``/``nstar``) are SOLVED, not
    # supplied, so they are never required here.
    _pspecs = model.get('parameters', []) or []
    _given = dict(fundamental) if fundamental else {}
    # (A saddle variable's default is never merged in: the legacy iteration
    # solver would take it as a given value and pin the saddle there.)
    fundamental = {p['name']: p['default'] for p in _pspecs
                   if p.get('default') is not None and not p.get('mean_field')}
    fundamental.update(_given)
    _missing = sorted(p['name'] for p in _pspecs
                      if p['name'] not in fundamental
                      and not p.get('mean_field'))
    if _missing:
        raise ValueError(
            'compute_cumulants: ' +
            ('no ``parameters=`` given' if not _given else
             'the ``parameters=`` given do not include them') +
            ', and these model parameters declare no default: ' +
            ', '.join(_missing) + '. '
            'Pass parameters={...} or declare defaults via '
            'ModelBuilder.parameter(name, default=...).')
    if external_fields is None:
        raise ValueError('external_fields is required')
    if len(external_fields) != k:
        raise ValueError(
            f'external_fields has {len(external_fields)} entries but k={k}'
        )
    if k < 1:
        raise ValueError(f'k must be >= 1; got {k}')
    if max_ell < 0:
        raise ValueError(f'max_ell must be >= 0; got {max_ell}')
    if backend is not None:
        from api import routing as _routing_check
        if backend not in {b.name for b in _routing_check.backends()}:
            raise ValueError(
                f'backend={backend!r} is not registered; known: '
                f'{[b.name for b in _routing_check.backends()]}')
    if tau_grid is None and not (tau_step > 0 and tau_max >= 0
                                 and np.isfinite(tau_step)
                                 and np.isfinite(tau_max)):
        raise ValueError(
            f'need finite tau_step > 0 and tau_max >= 0 (got tau_step='
            f'{tau_step}, tau_max={tau_max}), or pass an explicit tau_grid.')
    # A NaN lag is not "no lag": the Ito equal-time nudge turns a comparison
    # against it into a plain evaluation at tau = 0 and the result looks like
    # a valid number.  Refuse every grid / event array before any work.
    for _name, _arr in (('tau_grid', tau_grid), ('spatial_grid', spatial_grid),
                        ('spatial_points', spatial_points)):
        if _arr is None:
            continue
        _a = np.asarray(_arr, dtype=float)
        _bad = ~np.isfinite(_a)
        if _bad.any():
            _where = np.flatnonzero(_bad.ravel())[:5].tolist()
            raise ValueError(
                f'{_name} must be finite: {int(_bad.sum())} of {_a.size} '
                f'entries are NaN or infinite (flat positions {_where}'
                f'{"..." if _bad.sum() > 5 else ""}).')
    switch_on = _switch_on_request(
        model, k, max_ell, t_min, times, initial_covariance, tau_grid,
        spatial=(bool(model.get('spatial')) or spatial_grid is not None
                 or spatial_points is not None),
        grouped=use_grouped_phase_j,
        saving=bool(output_npz or output_csv))

    # Auto-pick a Taylor budget that covers every vertex the
    # prediagram enumerator could ask for at the chosen (k, max_ell).
    # Connected diagrams with k external legs at ell loops have at
    # most ``k + 2·ell``-leg vertices, so that's the tight upper
    # bound.  The floor of 2 is structural (anything less wouldn't
    # capture the (1,1) bilinear propagator + (0,0)/(1,0)/(0,1) MF
    # sectors that downstream code unconditionally reads).
    #
    # Historically this floor was 4 — chosen to keep all 1-loop
    # 2-cumulant runs in one cache directory back when the layout
    # was ``saved_models/<model>_taylor<N>/``.  The new layout
    # (``saved_models/<model>/expand_taylor<N>.sobj``) sibling-
    # files different orders cleanly, so the floor is no longer
    # needed for cache-directory cohesion.  Dropping it from 4 → 2
    # saves ~90 min on heavy Bernoulli models at ``k=2, max_ell=0``
    # (the only case where the old floor exceeded the math minimum).
    if taylor_order is None:
        taylor_order = max(k + 2 * max_ell, 2)

    # Accept user-facing natural names and translate to the internal
    # fluctuation names the action / propagator-typing code expect.
    # The mapping comes from ``model['naming_convention']`` (declared
    # by ModelBuilder); pipeline falls back to a classic n/v/m map
    # when the model doesn't declare its own.
    naming_convention   = model.get('naming_convention')
    external_fields_user = list(external_fields)
    external_fields      = normalize_external_fields(
        external_fields, naming_convention=naming_convention)

    # ── Per-phase wall-time tracker (verbose only) ─────────────────
    # Recorded into a dict so callers / notebooks can inspect the
    # breakdown via ``th['phase_walls']`` after the run, and so the
    # final `Done.` line can echo a summary.  Stays None when
    # verbose=False so the bookkeeping is opt-in.
    phase_walls: dict[str, float] | None = {} if verbose else None

    def _phase_time(label: str, t0: float) -> None:
        if phase_walls is None:
            return
        dt = time.perf_counter() - t0
        phase_walls[label] = dt
        print(f'      [{label}] done in {dt:.2f}s')

    # ── 0. Model analyzer gates (api.analyze; G0/G1 before the expansion) ──
    _pf = _make_preflight(preflight, model, k, max_ell, external_fields_user,
                          fundamental, fixed_point_index, policy, accuracy,
                          taylor_order)
    if _pf is not None:
        _pf.declarations()

    # ── 1. FieldTheory expansion (cache-aware) ────────────────────
    # Try the on-disk expand cache before paying the full
    # multivariate-taylor cost.  The cache stores ft._by_tp (the
    # bigrade-classified action dict) at each previously-computed
    # taylor_order; we accept any cached order >= the requested one
    # and downgrade-filter to total degree <= target.
    if verbose:
        print(f'[1/7] FieldTheory.expand (taylor_order={taylor_order})...')
    _t_phase = time.perf_counter()
    ft = FieldTheory(model, taylor_order=taylor_order)
    try:
        from api import _expand_cache as _ec
        cache_hit = False
        if use_cache:
            cached_order = _ec.find_best_cached_order(model, taylor_order)
            if cached_order is not None:
                _ec.prepare_for_load(ft)
                cache_hit = _ec.load_expand(
                    model, ft,
                    target_order=taylor_order,
                    cached_order=cached_order,
                    verbose=verbose,
                )
        if not cache_hit:
            ft.expand()
            if use_cache:
                try:
                    _ec.save_expand(model, ft, verbose=verbose)
                except Exception as e:
                    if verbose:
                        print(f'      [expand-cache] save failed ({e!r}); '
                              f'continuing without persistence.')

        sanity_ok = ft.sanity_check(verbose=verbose)
        if not sanity_ok:
            raise RuntimeError(
                'FieldTheory.sanity_check() failed — see printout above.'
            )
    except Exception as _exp_exc:
        if _pf is not None:               # e.g. no real saddle: name it
            _pf.failed('expanded', _exp_exc)
        raise
    if _pf is not None:
        _pf.expanded(ft)                  # G1: species, unpaired fields

    # ── 1b. Delay kernels: exp(-i omega d) in a kernel image (api.delay) ──
    if not model.get('_delay_pade') and model.get('kernel_ft_image') is not None:
        from api import delay as _delay
        _delays = _delay.kernel_delays(model, ft._ns)
        if _delays and switch_on is not None:
            # the delay driver's exact tree and its Pade-order search are
            # stationary (item 44: not combined yet)
            from engine.integration.time_domain import statespace as _ss_dl
            raise _ss_dl.SwitchOnUnsupported(
                'delay_kernels', 'models with delay kernels (api.delay) are '
                'computed for the stationary state only')
        if _delays:
            return _delay.compute_delayed(model, _delay_call, _delays)

    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)
    n_noise = sum(1 for s in stypes if isinstance(s, NoiseSourceType))
    if verbose:
        print(f'      vtypes: {len(vtypes)}, sources: {len(stypes)} '
              f'(NoiseSourceType: {n_noise})')
        _print_action_sectors(ft)
    _phase_time('expand', _t_phase)

    # ── 2. Propagator (symbolic, cached) ──────────────────────────
    if verbose:
        print('[2/7] Build propagator (K_ker → K_ft → G_ft → D_delta)...')
    _t_phase = time.perf_counter()
    prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)
    _phase_time('propagator', _t_phase)
    if _pf is not None:
        _pf.propagator(prop)              # G3: propagator class

    # ── 3. Mean-field solve + num_params assembly ─────────────────
    if verbose:
        print('[3/7] Solve MF self-consistency...')
    _t_phase = time.perf_counter()
    try:
        if model.get('equations'):
            # Route to the DAE-based multi-root solver when the model
            # declares ``.equation(lhs=..., rhs=..., population=...)``.
            # Returns the same legacy-shape keys plus DAE-specific extras
            # (mf_all_roots, mf_index_used, ...) that we surface on the
            # final th dict below.
            mf = solve_mean_field_dae_compat(
                ft, model, fundamental,
                fixed_point_index=fixed_point_index,
                n_starts=mf_dae_n_starts,
                seed_box=mf_dae_seed_box,
                verbose=verbose,
            )
        else:
            mf = solve_mean_field(ft, model, fundamental, verbose=verbose)
    except Exception as _mf_exc:
        if _pf is not None:               # name the cause when a gate can
            _pf.failed('mean_field', _mf_exc)
        raise
    num_params = mf['num_params']
    _phase_time('mean_field', _t_phase)
    if _pf is not None:
        _pf.mean_field(mf)                # G2 / G3 spatial / G4 / undriven

    # ── 3.5 Spatial short-circuit ─────────────────────────────────
    # A spatial model's propagator carries the inert ``Laplacian``
    # symbol, which the (ω) pole-finder and time-domain Phase J can't
    # consume.  Spatial models route to the full-diagram momentum
    # integrator instead (Symanzik ∫dᵈℓ → causal chambers → q→x FT),
    # which returns the real-space correlator C(x, τ).  See
    # docs/spatial_pipeline.md.  Requires spatial_grid (the x points).
    if spatial_grid is not None and not model.get('spatial'):
        import warnings as _warnings
        _warnings.warn(
            'spatial_grid was provided but the model is not spatial (no field '
            'declares spatial_dim>=1); it is ignored and the temporal C(τ) is '
            'returned.', stacklevel=2)

    # ── Routing (api.routing): the first registered backend that applies.
    # The built-in table reproduces the dispatch of 0.2.0 exactly (spatial
    # short-circuit, grouped / per-diagram Phase J) and raises the same
    # errors when nothing applies.
    from api import routing as _routing
    _request = _routing.RouteRequest(
        model_name=model.get('name', '<unnamed>'),
        spatial=bool(model.get('spatial')),
        k=k, max_ell=max_ell,
        has_spatial_grid=spatial_grid is not None,
        has_spatial_points=spatial_points is not None,
        grouped=bool(use_grouped_phase_j),
        initial_mode=('switch_on' if switch_on is not None else
                      (model.get('initial') or {}).get('mode', 'stationary')),
        requested=backend,
        traits=_pf.traits_dict() if _pf is not None else {})
    route = _routing.select(_request)
    if _pf is not None:
        _pf.route = route
    state = _PipelineState(
        model=model, k=k, max_ell=max_ell, ft=ft, prop=prop,
        num_params=num_params, mf=mf, external_fields=external_fields,
        external_fields_user=external_fields_user, fundamental=fundamental,
        spatial_points=spatial_points, spatial_grid=spatial_grid,
        tau_grid=tau_grid, tau_max=tau_max, tau_step=tau_step,
        taylor_order=taylor_order, origin_leaf_idx=origin_leaf_idx,
        use_cache=use_cache, parallel=parallel,
        spatial_parallel=spatial_parallel, spatial_n_q=spatial_n_q,
        n_workers=n_workers, use_grouped_phase_j=use_grouped_phase_j,
        verbose=verbose, naming_convention=naming_convention,
        vtypes=vtypes, stypes=stypes, phase_walls=phase_walls,
        phase_time=_phase_time, t_phase=_t_phase, preflight=_pf,
        route=route, evaluated_grid=False, switch_on=switch_on)
    _snap = _counter_snapshot()
    try:
        result = route.backend.run(state)
    finally:                              # restores the maxima on a raise too
        _counters, _complete = _counter_delta(_snap)
    # S3: the Phase J engine census of the run (subsets on the state-space
    # route, declined by reason, call-time fallbacks by reason)
    if isinstance(result, dict) and 'phase_j_by_ell' in result:
        _engine_census(result, _counters)
    # Never silent: a diagram Phase J could not evaluate is named (status,
    # warning; strict policy raises after the report is attached).
    _failures = None
    if isinstance(result, dict) and 'phase_j_by_ell' in result:
        _failures = _phase_j_failures(result['phase_j_by_ell'])
        result['phase_j_incomplete'] = _failures
        result['status'] = 'incomplete' if _failures else 'ok'
    if _pf is not None:
        # G6 sizes its accuracy estimate on ``C_tau``; a switch-on run's
        # values are ``C_times`` (a read-only view: G6 never writes)
        _pf.after_run(
            dict(result, C_tau=result.get('C_times'))
            if switch_on is not None and isinstance(result, dict)
            else result, _counters, complete=_complete,
            evaluated=state.evaluated_grid)
    _attach_provenance(result, route, _pf)
    if isinstance(result, dict) and isinstance(result.get('route'), dict) \
            and result.get('phase_j_engine') is not None:
        result['route']['phase_j_engine'] = result['phase_j_engine']
    if switch_on is not None and isinstance(result, dict):
        _switch_on_check(result, _failures, model.get('name', '<unnamed>'))
    if _failures:
        _msg = _phase_j_failure_message(_failures,
                                        model.get('name', '<unnamed>'))
        if _policy_is_strict(policy):
            raise PhaseJIncompleteError(_msg, result=result,
                                        failures=_failures)
        # visible under ``sage -python`` / the Sage kernel too, whose
        # launcher sets PYTHONWARNINGS=ignore
        from api.save import _ensure_warning_shown
        _ensure_warning_shown(PhaseJIncompleteWarning, 'always')
        warnings.warn(_msg, PhaseJIncompleteWarning, stacklevel=2)
    if route.backend.standard_tail:
        _standard_tail(state, result, output_npz, output_csv)
    return result


def _run_spatial(state):
    """Backend ``spatial_integrator`` (``api.routing``): the full-diagram
    momentum integrator.  ``state``: the pipeline objects after the
    mean-field stage (``_PipelineState``); the applicability checks
    (k = 2 needs ``spatial_grid``, k != 2 ``spatial_points``, ``max_ell <=
    2``, stationary initial state) are the backend's ``applies``."""
    (model, k, max_ell, ft, prop, num_params, mf, external_fields,
     spatial_points, spatial_grid, tau_grid, tau_max, tau_step, verbose,
     parallel, spatial_parallel, n_workers, spatial_n_q, fundamental,
     naming_convention, _phase_time, _t_phase) = (
        state.model, state.k, state.max_ell, state.ft, state.prop,
        state.num_params, state.mf, state.external_fields,
        state.spatial_points, state.spatial_grid, state.tau_grid,
        state.tau_max, state.tau_step, state.verbose, state.parallel,
        state.spatial_parallel, state.n_workers, state.spatial_n_q,
        state.fundamental, state.naming_convention, state.phase_time,
        state.t_phase)
    import numpy as _np
    from engine.integration.spatial.pipeline_bridge import (
        compute_spatial_correlator_via_pipeline,
    )
    from engine.integration.spatial.spatial_correlator import (
        free_two_point,
    )
    if k != 2:
        # ── general-k spatial path: explicit evaluation events ──
        from engine.integration.spatial.pipeline_bridge import (
            compute_spatial_kpoint,
        )
        try:
            vals, kp_info = compute_spatial_kpoint(
                ft, model, prop, num_params, external_fields,
                spatial_points, max_ell=max_ell, verbose=verbose)
        except NotImplementedError as _e:
            if 'single-field' not in str(_e):
                raise
            # ── COUPLED multi-field theory at k>=3: the spectral-
            # assignment driver (scalar diffusion, plain vertices).
            from engine.integration.spatial.heat_kernel import (
                reaction_diffusion_matrices)
            from engine.integration.spatial.pipeline_bridge import (
                compute_coupled_kpoint, _norm_sr)
            from engine.integration.spatial.spectral_propagator import (
                split_reference_diffusion)
            _ns = ft._ns
            _Msr, _Dsr, _Vsr = reaction_diffusion_matrices(
                prop['K_ft'], prop['omega'], SR.var('k'),
                _ns.Laplacian, getattr(_ns, 'GradX', None))
            _nf = int(_Msr.nrows())
            _sub = _norm_sr(num_params)

            def _numm(sm):
                return _np.array(
                    [[complex(SR(sm[a_, b_]).subs(_sub)).real
                      for b_ in range(_nf)] for a_ in range(_nf)],
                    dtype=float)
            _Mn, _Dn, _Vn = _numm(_Msr), _numm(_Dsr), _numm(_Vsr)
            if not _np.allclose(_Vn, 0.0, atol=1e-9):
                raise NotImplementedError(
                    'coupled k>=3: drift (V != 0) not supported.')
            _D0, _Dhat = split_reference_diffusion(_Dn)
            tree_info_k = {
                'M': _Mn, 'Dhat': _Dhat, 'D0': float(_D0),
                'bc_mode': (model.get('boundary')
                            or {}).get('mode', 'infinite')}
            vals, kp_info = compute_coupled_kpoint(
                ft, model, prop, num_params, external_fields,
                spatial_points, tree_info_k, max_ell=max_ell,
                verbose=verbose)
        _phase_time('spatial-kpoint', _t_phase)
        return {
            'C_kpoint':        vals,
            'C_kpoint_by_ell': kp_info['per_ell'],
            'points':          _np.asarray(spatial_points, dtype=float),
            'spatial_info':    kp_info,
            'k': k, 'max_ell': max_ell,
            'mf':              mf,
        }
    tau_grid = (_np.asarray(tau_grid, dtype=float) if tau_grid is not None
                else _np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step))
    spatial_grid_arr = _np.asarray(spatial_grid, dtype=float)
    if verbose:
        print('[4/7] (spatial) Momentum stays symbolic (Laplacian) — skip the '
              'ω pole-finder + time-domain Phase J; route to the full-diagram '
              'integrator.')
        print(f'[spatial] {"loop" if max_ell >= 1 else "tree-level"} '
              f'C(x, τ): max_ell={max_ell}, {len(tau_grid)} τ × '
              f'{len(spatial_grid_arr)} x points...')
    # Route through the SHARED pipeline (Stage B/C): the bridge runs the
    # real diagram pipeline at sample momenta to CERTIFY the per-mode
    # (A,B,N) structure, then does the q→x FT.  At max_ell≥1 the FULL-DIAGRAM
    # integrator sums EVERY enumerated diagram up to max_ell loops through one
    # genuine integral (Symanzik ∫dᵈℓ → causal-chamber time integral →
    # ret+adv), weighted by the enumeration 𝒮(Γ) — no shortcut, no diagram
    # dropped (docs/spatial_generic_pipeline_plan.md).
    if max_ell >= 1:
        from engine.integration.spatial.pipeline_bridge import (
            compute_spatial_correlator_generic,
        )
        # Spatial parallelism is THREAD-based (no fork → safe in Jupyter/macOS;
        # numpy releases the GIL for the per-diagram linalg/quadrature).  On by
        # default; disable with spatial_parallel=False.  (Fork is forbidden here.)
        C_tau_x, sp_info = compute_spatial_correlator_generic(
            ft, model, prop, num_params, external_fields,
            tau_grid, spatial_grid_arr, verbose=verbose, max_ell=max_ell,
            parallel=(parallel and spatial_parallel), n_workers=n_workers,
            n_q=int(spatial_n_q),
        )
    else:
        C_tau_x, sp_info = compute_spatial_correlator_via_pipeline(
            ft, model, prop, num_params, external_fields,
            tau_grid, spatial_grid_arr, verbose=verbose, stage_headers=True,
        )
    if verbose:
        _cmr = sp_info.get("certify_max_rel")
        _cmr_s = f'{_cmr:.1e}' if _cmr is not None else 'n/a (coupled driver)'
        print(f'[spatial] done — C(x,τ) ready; tree-mode certified='
              f'{sp_info.get("pipeline_certified")} '
              f'(max rel {_cmr_s})')
    # x=0 slice as the conventional C_tau (matches the time-only
    # API's C_tau shape).
    x0_idx = int(_np.argmin(_np.abs(spatial_grid_arr)))
    C_tau = C_tau_x[:, x0_idx].copy()

    sp_d = int(sp_info.get('spatial_dim', 1))

    def total_C(*tau_then_x):
        """C(τ) at x=0 (1 arg) or C(x, τ) (2 args: τ, x)."""
        if len(tau_then_x) == 1:
            tau = float(tau_then_x[0])
            xq = 0.0
        else:
            tau, xq = float(tau_then_x[0]), float(tau_then_x[1])
        if sp_d == 1:
            val = 0j
            for (A, B, N) in sp_info['modes']:
                val += free_two_point(A, B, N, xq, tau,
                                      bc_mode=sp_info['bc_mode'],
                                      L=sp_info['L'])
            # 1-loop tadpole mass-shift correction δC = Σ·∂C₀/∂A.
            if sp_info.get('Sigma'):
                A0, B0, N0 = sp_info['modes'][0]
                A0 = float(_np.real(A0))
                h = 1e-4 * max(1.0, abs(A0))
                fp = free_two_point(A0 + h, B0, N0, xq, tau,
                                    bc_mode=sp_info['bc_mode'], L=sp_info['L'])
                fm = free_two_point(A0 - h, B0, N0, xq, tau,
                                    bc_mode=sp_info['bc_mode'], L=sp_info['L'])
                val += sp_info['Sigma'] * (fp - fm) / (2.0 * h)
            return val
        # d≥2: radial q→x transform of the momentum-space correlator
        # (+ optional tadpole mass-shift via a finite difference in A).
        from engine.integration.spatial.spatial_correlator import (
            radial_inverse_ft,
        )
        qg = _np.linspace(40.0 / 8000.0, 40.0, 2000)
        at = abs(tau)
        Cq = _np.zeros_like(qg)
        for (A, B, N) in sp_info['modes']:
            A = float(_np.real(A)); B = float(_np.real(B)); N = float(_np.real(N))
            m = A + B * qg * qg
            Cq += (N / m) * _np.exp(-m * at)
        if sp_info.get('Sigma'):
            A0, B0, N0 = sp_info['modes'][0]
            A0 = float(_np.real(A0)); B0 = float(_np.real(B0)); N0 = float(_np.real(N0))
            h = 1e-4 * max(1.0, abs(A0))
            mp = (A0 + h) + B0 * qg * qg
            mm = (A0 - h) + B0 * qg * qg
            Cq += sp_info['Sigma'] * (
                (N0 / mp) * _np.exp(-mp * at) - (N0 / mm) * _np.exp(-mm * at)
            ) / (2.0 * h)
        return complex(radial_inverse_ft(qg, Cq, abs(xq), sp_d))

    # MF values (for the result dict) — reuse the saddle solve.
    mf_values_sp = {}
    for pname, vals in (mf.get('mf_values') or {}).items():
        mf_values_sp[pname] = vals

    return {
        'total_C':        total_C,
        'total_C_by_ell': {0: total_C},
        'C_tau':          C_tau,
        'C_tau_x':        C_tau_x,
        # cumulative C(x,τ) at each loop order {0: tree, 1: +1-loop, …, max_ell}
        # — the WHOLE progression from ONE call (the highest-ℓ run already
        # contains every lower order; no need to re-run per order).
        'C_tau_x_by_order': sp_info.get('C_by_order', {0: C_tau_x}),
        'tau_grid':       tau_grid,
        'spatial_grid':   spatial_grid_arr,
        'chi_grid':       spatial_grid_arr,   # canonical alias of spatial_grid
        'spatial_info':   sp_info,
        'mf_values':      mf_values_sp,
        'mf':             MeanField(mf_values_sp,
                                    naming_convention=naming_convention),
        'params':         Parameters(fundamental),
        'num_params':     num_params,
        'propagator':     prop,
        'config': {
            'k':               k,
            'max_ell':         max_ell,
            'fundamental':     fundamental,
            'parameters':      fundamental,
            'external_fields': external_fields,
            'tau_max':         tau_max,
            'tau_step':        tau_step,
            'spatial':         True,
            'model_name':      model.get('name', '<unnamed>'),
            'phase_j_convention': _phase_j_convention(),
            'phase_j_flags':      _phase_j_flags(),
        },
    }


def _run_phase_j_per_diagram(state):
    """Backend ``phase_j_per_diagram`` (``api.routing``)."""
    return _run_phase_j(state, grouped=False)


def _run_phase_j_grouped(state):
    """Backend ``phase_j_grouped`` (``api.routing``)."""
    return _run_phase_j(state, grouped=True)


def _run_phase_j(state, grouped):
    """The temporal Phase J stages [4/7]-[7/7]: poles and residues, the
    diagram enumeration, the prefactor classification and the time-domain
    integration (``grouped``: the grouped integrator), and the result dict.
    ``state``: the pipeline objects after the mean-field stage."""
    (model, k, max_ell, ft, prop, num_params, mf, external_fields,
     external_fields_user, fundamental, tau_grid, tau_max, tau_step,
     taylor_order, origin_leaf_idx, use_cache, parallel, n_workers,
     verbose, naming_convention, vtypes, stypes, phase_walls,
     _phase_time) = (
        state.model, state.k, state.max_ell, state.ft, state.prop,
        state.num_params, state.mf, state.external_fields,
        state.external_fields_user, state.fundamental, state.tau_grid,
        state.tau_max, state.tau_step, state.taylor_order,
        state.origin_leaf_idx, state.use_cache, state.parallel,
        state.n_workers, state.verbose, state.naming_convention,
        state.vtypes, state.stypes, state.phase_walls, state.phase_time)
    use_grouped_phase_j = bool(grouped)
    # ── 4. Numerical poles + residues (fills prop in place) ───────
    if verbose:
        print('[4/7] Compute numerical poles + residue matrices...')
    _t_phase = time.perf_counter()
    compute_poles_and_residues(prop, num_params, verbose=verbose,
                               tau_max=_tau_extent(tau_grid, tau_max),
                               resolve_clusters=True)
    _phase_time('poles', _t_phase)
    # S3: whether the state-space route takes the whole run (no noise-kernel
    # sources, every conductance kernel realizable, a Hurwitz realization)
    ss_route = _ss_route_prediction(
        prop, num_params, list(vtypes or ()) + list(stypes or ()))
    switch_on = getattr(state, 'switch_on', None)
    if switch_on is not None:
        if not ss_route.get('covers_run'):
            from engine.integration.time_domain import statespace as _ss_so
            raise _ss_so.SwitchOnUnsupported(
                ss_route.get('reason') or 'not_covered',
                'the run-level prediction of the state-space route')
        _ledger_switch_on(state.preflight, switch_on)
    if state.preflight is not None:
        # G3 pole statistics (api.analyze); informational when the
        # state-space route covers the run
        state.preflight.poles(prop, ss_route=ss_route)

    # ── 5. Field-index maps + prediagram + typed diagram pipeline ─
    if verbose:
        print(f'[5/7] Enumerate prediagrams (k={k}, max_ell={max_ell})...')
    _t_phase = time.perf_counter()
    ring_var_names = list(ft._ns._ring_var_names)
    n_tilde = ft._n_tilde
    resp_idx, phys_idx = build_field_index_map(ring_var_names, n_tilde)
    for f in external_fields:
        if f not in phys_idx:
            raise ValueError(
                f'external field {f} not in phys_idx '
                f'{sorted(phys_idx.keys())}'
            )

    unique_by_ell, multiplicity_by_ell, all_unique = enumerate_unique_diagrams(
        ft, model,
        k               = k,
        max_ell         = max_ell,
        external_fields = external_fields,
        G_ft            = prop['G_ft'],
        resp_idx        = resp_idx,
        phys_idx        = phys_idx,
        vtypes          = vtypes,
        stypes          = stypes,
        use_cache       = use_cache,
        # Cache off: the eager records, not the streamed ones, until Phase
        # J's nquad fallback is representative-independent (see the flag).
        stream_prediagrams = _pd_cache.TEMPORAL_CACHE_OFF_STREAMS,
        parallel        = parallel,
        n_workers       = n_workers,
        verbose         = verbose,
    )
    _phase_time('diagrams', _t_phase)

    # ── 6. Diagram-level prefactor classification + kernel groups ─
    if verbose:
        print('[6/7] Classify coefficient factors per diagram...')
    _t_phase = time.perf_counter()
    time_dep_params = model.get('time_dependent_parameters', []) or []
    noise_structure = model.get(
        'noise_structure', {'temporal_type': 'white', 'amplitude_params': []}
    )

    diagram_records = []
    kernel_groups = []
    # Walk by ell so each record carries the ell tag — needed for the
    # per-loop-order Phase J decomposition in step [7].
    for ell in sorted(unique_by_ell.keys()):
        mults = multiplicity_by_ell.get(ell, [1] * len(unique_by_ell[ell]))
        for td, mult in zip(unique_by_ell[ell], mults):
            info = classify_coefficient_factors(
                td, time_dep_params, noise_structure
            )
            # Path A: ``combinatorial_factor`` in symmetry.py now
            # computes 𝒮(Γ) = ∏ n_leg! / |Aut_fixed_ext(Γ)| using the
            # full coloured incidence-graph automorphism order.  That
            # already accounts for every Feynman-rule symmetry —
            # same-type vertex swaps, parallel-edge swaps, self-loop
            # leg swaps — so multiplying by ``mult`` (the dedup
            # equivalence-class size) here would double-count and is
            # disabled.
            combined_prefactor = SR(info['scalar_prefactor'])
            kernel_groups.append({
                'diagrams':           [td],
                'combined_prefactor': combined_prefactor,
            })
            diagram_records.append({
                'typed_diagram':      td,
                'classify':           info,
                'combined_prefactor': combined_prefactor,
                # M(F) in the resolved convention of the paper's
                # Remark rem_multiplicity_convention: prod n_leg! /
                # |Aut_ext|, the weight that multiplies the literal
                # vertex coefficients (combined_prefactor = M * coeffs).
                'multiplicity':       info['Scal'],
                # size of the deduplication class the representative
                # stands for — diagnostic only, NOT a weight (it used
                # to be stored under 'multiplicity', which the TikZ
                # panel captions then printed as M).
                'dedup_class_size':   mult,
                'ell':                ell,
            })
    _phase_time('classify', _t_phase)

    # ── 7. Phase J time-domain integration (per ell) ──────────────
    if verbose:
        print(f'[7/7] Phase J: compute_correction_td per ell '
              f'(0..{max_ell})...')
    _t_phase = time.perf_counter()
    tau_grid = (np.asarray(tau_grid, dtype=float) if tau_grid is not None
                else np.arange(-tau_max, tau_max + tau_step * 0.5, tau_step))

    propagator_data = {
        'K_ker':   prop['K_ker'],
        'K_ft':    prop['K_ft'],
        'G_ft':    prop['G_ft'],
        'adj_ft':  prop['adj_ft'],
        'D_omega': prop['D_omega'],
        'D_delta': prop['D_delta'],
        't_var':   prop['t_var'],
        'omega':   prop['omega'],
        'nf':      prop['nf'],
        'pole_vals': prop['pole_vals'],
        'C_mats':    prop['C_mats'],
        # Identifies the model in Phase J's warnings (the hardened
        # quadrature fallback warns once per model, source and loop order).
        'model_name': model.get('name'),
    }

    # Item 35: the memory pre-pass, before any modesum plan is built.
    mem_estimate = _phase_j_memory_preflight(
        diagram_records, propagator_data, num_params, prop, model, k,
        max_ell, verbose, state.preflight, ss_route=ss_route,
        n_tau=(len(tau_grid) if k <= 2 else 1) if switch_on is None
        else len(switch_on['times']))

    # Pick the τ-grid evaluation pattern by k (None = no grid eval)
    if k == 1:
        tau_points = [(float(t),) for t in tau_grid]
    elif k == 2:
        # Single-axis slice: vary leaf 1 over tau_grid, leaf 0 pinned at 0.
        # Itô LEFT-limit at equal time (Itô ⇒ left-continuous, Θ(0)=Θ(0⁻)): the
        # τ=0 entry is sampled at t_1 = −_ITO_EPS rather than exactly 0.  The
        # propagators are strictly-causal retarded with Θ(0)=0; landing the τ-grid
        # on that measure-zero step discontinuity drops a coincident-ordering
        # contribution and produces a spurious single-point dip at τ=0.  Nudging
        # to 0⁻ keeps the surviving causal chamber and gives the left limit (= the
        # right limit for an even/stationary C; differs only at a genuine τ=0
        # jump, where Itô takes the left).  The grid LABEL stays 0.
        tau_points = [(0.0, float(t) if abs(float(t)) > 1e-12 else -_ITO_EPS)
                      for t in tau_grid]
    else:
        tau_points = None   # k≥3: caller handles grid evaluation
    if switch_on is not None:
        # absolute times (``times``), evaluated below; no τ grid
        tau_points = None
        so_points = [tuple(float(x) for x in row)
                     for row in switch_on['times']]

    # Run Phase J once per ell so we get the per-loop-order
    # decomposition the saver / notebook plots need.  Diagrams within
    # an ell are summed in that ell's total_C; the master total_C is
    # the sum across ells.  One pass runs every ell on ``pdata``;
    # near-degenerate or repeated poles (``prop['pole_deformation']``)
    # average passes over a deformation of the propagator
    # (``api._pole_clusters``).
    def _phase_j_pass(pdata):
        phase_j_by_ell  = {}
        total_C_by_ell  = {}
        C_tau_by_ell    = {}
        for ell in sorted(unique_by_ell.keys()):
            records_ell = [r for r in diagram_records if r['ell'] == ell]
            if not records_ell:
                # No diagrams at this ell (e.g. tree-level for some
                # subset configs).  Contribute zero.
                total_C_by_ell[ell] = (lambda *t: 0.0 + 0.0j)
                phase_j_by_ell[ell] = None
                if tau_points is not None:
                    C_tau_by_ell[ell] = np.zeros(len(tau_grid), dtype=complex)
                elif switch_on is not None:
                    C_tau_by_ell[ell] = np.zeros(len(so_points), dtype=complex)
                else:
                    C_tau_by_ell[ell] = None
                continue

            if use_grouped_phase_j:
                # Prototype: group typed diagrams by parent prediagram and
                # sum integrands before fast_callable + quadrature.  See
                # ``api/_grouped_phase_j.py`` for the math + caveats.
                from api._grouped_phase_j import (
                    compute_correction_td_grouped,
                )
                td_result_ell = compute_correction_td_grouped(
                    typed_diagrams   = [r['typed_diagram']
                                        for r in records_ell],
                    prefactors       = [r['combined_prefactor']
                                        for r in records_ell],
                    k                = k,
                    propagator_data  = pdata,
                    external_fields  = external_fields,
                    num_params       = num_params,
                    origin_leaf_idx  = origin_leaf_idx,
                )
            else:
                td_result_ell = compute_correction_td(
                    typed_diagrams   = [r['typed_diagram']
                                        for r in records_ell],
                    prefactors       = [r['combined_prefactor']
                                        for r in records_ell],
                    k                = k,
                    propagator_data  = pdata,
                    external_fields  = external_fields,
                    num_params       = num_params,
                    origin_leaf_idx  = origin_leaf_idx,
                )
            if switch_on is not None:
                total_C_by_ell[ell] = _phase_j_scoped(_switch_on_callable(
                    td_result_ell['total_C'], switch_on['t_min']))
            else:
                total_C_by_ell[ell] = _phase_j_scoped(
                    _ito_nudge_callable(td_result_ell['total_C'], k))
            phase_j_by_ell[ell] = td_result_ell

            if switch_on is not None:
                with _phase_j_warning_scope():
                    C_tau_by_ell[ell] = np.array(
                        [complex(total_C_by_ell[ell](*pt))
                         for pt in so_points], dtype=complex)
                _switch_on_wrap_td(td_result_ell, switch_on['t_min'])
            elif tau_points is not None:
                with _phase_j_warning_scope():
                    C_tau_by_ell[ell] = np.array(
                        td_result_ell['total_C_batch'](
                            tau_points, parallel=parallel, n_workers=n_workers),
                        dtype=complex,
                    )
            else:
                C_tau_by_ell[ell] = None
        return phase_j_by_ell, total_C_by_ell, C_tau_by_ell

    pole_cluster_report = _pole_cluster_preamble(prop, max_ell, ss_route)
    engine_report = dict(ss_route, pole_deformation=(
        None if prop.get('pole_deformation') is None
        else 'bypassed' if ss_route['bypass_deformation'] else 'used'))
    # the running guard: every plan of this run counts against one budget
    from engine.integration.time_domain import final_integral as _fi_mem
    import contextlib as _ctx
    _scope = (_fi_mem.phase_j_memory_scope(
        what=f"{model.get('name', '<unnamed>')} k={k} max_ell={max_ell}")
        if hasattr(_fi_mem, 'phase_j_memory_scope')       # older engine: A/B
        else _ctx.nullcontext())
    if switch_on is not None:
        from engine.integration.time_domain import statespace as _ss_so
        _so_scope = _ss_so.switch_on_scope(switch_on['t_min'])
    else:
        _so_scope = _ctx.nullcontext()
    with _scope as _mem_scope, _so_scope:
        _bypassed = None
        if prop.get('pole_deformation') is not None and (
                ss_route['bypass_deformation'] or switch_on is not None):
            _bypassed = _phase_j_ss_bypass(_phase_j_pass, propagator_data,
                                           engine_report)
            if _bypassed is None and switch_on is not None:
                # the deformation averages pole-engine passes
                from engine.integration.time_domain import statespace as _ssd
                raise _ssd.SwitchOnUnsupported(
                    'pole_deformation', engine_report.get('bypass_failed'))
        if _bypassed is not None:
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _bypassed
        elif prop.get('pole_deformation') is None:
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _phase_j_pass(
                propagator_data)
        else:
            if pole_cluster_report is not None and pole_cluster_report.get(
                    'status') == 'bypassed':
                # the bypass did not hold (a subset needed the pole route):
                # the deformation runs after all
                pole_cluster_report = _pole_cluster_preamble(prop, max_ell)
            phase_j_by_ell, total_C_by_ell, C_tau_by_ell = _phase_j_deformed(
                _phase_j_pass, propagator_data, prop['pole_deformation'],
                diagram_records, num_params, tau_points, pole_cluster_report)
    if mem_estimate is not None and _mem_scope is not None:
        mem_estimate['built_tuples'] = _mem_scope.tuples
        mem_estimate['built_bytes'] = _mem_scope.used
        mem_estimate['built_plans'] = _mem_scope.plans

    _phase_time('phase_j', _t_phase)

    if use_grouped_phase_j:
        # M7: say so once per run if the grouped path dropped shot-noise
        # δ(τ) records (since Feature B it carries them: never fires).
        from api._grouped_phase_j import warn_dropped_delta_records
        warn_dropped_delta_records(phase_j_by_ell)

    # Master total_C: sum across ell (for caller convenience)
    @_phase_j_scoped
    def total_C(*ext_time_values):
        return sum(complex(fn(*ext_time_values))
                   for fn in total_C_by_ell.values())
    # item 35 (F3): ``dd.run`` may form total_C's values from the per-ell
    # values it evaluates anyway (the same sum, in this order)
    total_C.sum_of_by_ell = total_C_by_ell

    # Aggregate τ-grid C(τ) = Σ_ell C_ell(τ)
    if tau_points is not None:
        C_tau = sum(C_tau_by_ell[ell] for ell in C_tau_by_ell)
    else:
        C_tau = None
    so_record = None
    if switch_on is not None:
        # switch-on: the values at ``times`` (C_tau_by_ell holds them); the
        # tree-level initial-state term joins ell = 0
        C_times_by_ell = C_tau_by_ell
        C_tau_by_ell = {ell: None for ell in C_times_by_ell}
        so_record = {
            'protocol': 'switch_on', 't_min': switch_on['t_min'],
            'engine': 'statespace',
            # Markov-embedded noise kernels (auxiliary fields made when the
            # model was built): their states start at zero as well
            'embedded_noise_states': _embedded_noise_states(model),
            'initial_covariance': switch_on['initial_covariance'],
            'initial_term': None}
        init = _initial_state_term(switch_on, propagator_data, num_params,
                                   external_fields, phys_idx)
        if init is not None:
            init_vals, init_fn = init
            so_record['initial_term'] = init_vals
            C_times_by_ell[0] = C_times_by_ell.get(0, 0) + init_vals
            _tree = total_C_by_ell.get(0, lambda *t: 0.0 + 0.0j)
            total_C_by_ell[0] = (lambda *t, _f=_tree, _g=init_fn:
                                 complex(_f(*t)) + _g(*t))
            # the ell = 0 Phase J record's totals too (its per-diagram
            # closures, ``groups`` / ``eval_per_diagram_batch``, are the
            # diagrams alone: the term is not a diagram)
            _td0 = phase_j_by_ell.get(0)
            if isinstance(_td0, dict) and callable(_td0.get('total_C')):
                _f0 = _td0['total_C']
                _td0['total_C'] = (lambda *t, _f=_f0, _g=init_fn:
                                   complex(_f(*t)) + _g(*t))
                _td0['total_C_batch'] = (
                    lambda tau_points, *a, _f=_td0['total_C'], **kw:
                    [complex(_f(*p)) for p in tau_points])
        C_times = sum(C_times_by_ell[ell] for ell in C_times_by_ell)

    # ── Adaptive mean-field dict ──────────────────────────────────
    # Discover MF saddle quantities from the model itself.  Two paths:
    #   1. ``model['naming_convention']['mf_parameters']`` (preferred,
    #      written by ModelBuilder when ``mean_field=True`` is set)
    #   2. ``parameters[i]['mean_field']`` flag (the same data)
    #   3. Classic n/v/m fallback for hand-written model files that
    #      pre-date the declarative API.
    npop = len(ft._ns.pop)

    mf_param_names: list[str] = []
    if naming_convention and naming_convention.get('mf_parameters'):
        mf_param_names = list(naming_convention['mf_parameters'])
    else:
        # Walk model['parameters'] looking for the flag
        for pspec in model.get('parameters', []) or []:
            if pspec.get('mean_field'):
                mf_param_names.append(pspec['name'])
        # Last-resort fallback for legacy models
        if not mf_param_names:
            for legacy in ('nstar', 'vstar', 'mstar'):
                if hasattr(ft._ns, legacy):
                    mf_param_names.append(legacy)

    # Map each MF param name to the range of local indices it owns.
    # Heterogeneous models declare ``indexed_by=['<pop>']`` on each
    # saddle; the param's SR array is sized to ``len(pop_<pop>)``.
    # Legacy single-pop models use the flat ``ns.pop`` length.
    param_specs_by_name = {
        pspec['name']: pspec
        for pspec in (model.get('parameters', []) or [])
    }
    pop_size_map = getattr(ft._ns, '_pop_size', {}) or {}

    def _saddle_indices(pname):
        pspec = param_specs_by_name.get(pname, {})
        ib = pspec.get('indexed_by') or []
        if ib:
            # Heterogeneous: iterate over the saddle's own population
            # (single-population saddles only — physical_fields are
            # currently restricted to one population per field).
            n = pop_size_map.get(ib[0], 0)
            return list(range(n))
        # Legacy: flat pop index
        return list(ft._ns.pop)

    mf_values: dict[str, list] = {}
    for pname in mf_param_names:
        if not hasattr(ft._ns, pname):
            continue
        ns_syms = getattr(ft._ns, pname)
        # ns_syms may be a single SR var (un-indexed param) or a list
        if not isinstance(ns_syms, (list, tuple)):
            ns_syms = [ns_syms]
        vals: list[float] = []
        for i in _saddle_indices(pname):
            if i >= len(ns_syms):
                vals.append(float('nan'))
                continue
            sym = ns_syms[i]
            if sym in num_params:
                vals.append(float(num_params[sym]))
            else:
                vals.append(float('nan'))
        # Drop entries that are entirely NaN (model declared the
        # symbol but the MF solver never substituted — e.g. mstar
        # in a non-GTaS model that still ships an mstar declaration).
        if not all(v != v for v in vals):
            mf_values[pname] = vals

    result = {
        # S3: the Phase J engine of the run (prediction, deformation; the
        # per-subset census is added by compute_cumulants)
        'phase_j_engine':  engine_report,
        'phase_j_memory':  mem_estimate,
        'total_C':         total_C,
        'total_C_by_ell':  total_C_by_ell,
        'C_tau':           C_tau,
        'C_tau_by_ell':    C_tau_by_ell,
        'tau_grid':        tau_grid if switch_on is None else None,
        'mf_values':       mf_values,
        'mf':              MeanField(mf_values,
                                     naming_convention=naming_convention),
        'params':          Parameters(fundamental),
        'num_params':      num_params,
        'propagator':      prop,
        'diagrams':        diagram_records,
        'kernel_groups':   kernel_groups,
        'phase_j_by_ell':  phase_j_by_ell,
        'phase_walls':     phase_walls,  # dict[label → seconds], None if not verbose
        # near-degenerate / repeated poles (api._pole_clusters): None when no
        # two poles are close, else the cluster report (status, accuracy)
        'pole_clusters':   pole_cluster_report,
        'config': {
            'k':                  k,
            'max_ell':            max_ell,
            'fundamental':        fundamental,
            'parameters':         fundamental,
            'external_fields':    external_fields,        # internal names
            'external_fields_in': external_fields_user,   # as user passed
            'tau_max':            tau_max,
            'tau_step':           tau_step,
            'taylor_order':       taylor_order,
            'model_name':         model.get('name', '<unnamed>'),
            # Read after the τ-grid evaluation above: the rule these numbers
            # were computed with (the Phase J flags are read at call time).
            'phase_j_convention': _phase_j_convention(),
            'phase_j_flags':      _phase_j_flags(use_grouped_phase_j),
            'use_grouped_phase_j': bool(use_grouped_phase_j),
        },
    }

    # ── DAE-MF extras (when the new equation-based solver ran) ────
    # ``mf_all_roots`` carries every distinct root with a per-root
    # stability annotation (``stable: bool``, ``eigenvalues_finite``).
    # ``fixed_point_index`` indexes over the STABLE subset only —
    # ``mf_stable_roots`` is that subset in the same sort order.
    # Unstable roots are surfaced via ``mf_unstable_roots`` for
    # inspection but are not selectable as expansion points.
    if 'mf_all_roots' in mf:
        result['mf_all_roots']        = mf['mf_all_roots']
        result['mf_stable_roots']     = mf.get('mf_stable_roots', [])
        result['mf_unstable_roots']   = mf.get('mf_unstable_roots', [])
        result['mf_index_used']       = mf['mf_index_used']
        result['mf_state_var_order']  = mf['state_var_order']
        # The selected root is guaranteed stable; its stability dict
        # (with full eigenvalue spectrum + A/B matrices) is the
        # canonical record for downstream consumers.
        try:
            result['mf_stability'] = linear_stability(
                model, fundamental, mf['mf_values'], verbose=False)
        except Exception as e:
            if verbose:
                print(f'  [stability] skipped: {type(e).__name__}: {e}')
            result['mf_stability'] = None
    if switch_on is not None:
        result['times'] = switch_on['times']
        result['C_times'] = C_times
        result['C_times_by_ell'] = C_times_by_ell
        result['t_min'] = switch_on['t_min']
        result['nonstationary'] = so_record
        result['config']['t_min'] = switch_on['t_min']
        result['config']['times'] = switch_on['times']
    state.all_unique = all_unique
    state.tau_grid_eval = (tau_grid if switch_on is None
                           else switch_on['times'][:, 0])
    state.evaluated_grid = tau_points is not None or switch_on is not None
    return result


def _standard_tail(state, result, output_npz, output_csv):
    """Save ``output_npz`` / ``output_csv`` and print the summary (the
    backends with ``standard_tail``: the temporal Phase J ones)."""
    verbose, phase_walls = state.verbose, state.phase_walls
    k, max_ell = state.k, state.max_ell
    all_unique, tau_grid = state.all_unique, state.tau_grid_eval
    # ── Optional NPZ / CSV save (api.save handles the schema) ─
    if output_npz or output_csv:
        from api.save import save_npz, save_csv
        if output_npz:
            save_npz(result, output_npz)
            if verbose:
                print(f'\nSaved NPZ: {output_npz}')
        if output_csv:
            save_csv(result, output_csv)
            if verbose:
                print(f'\nSaved CSV: {output_csv}')

    if verbose:
        print(f'\nDone.  k={k}, max_ell={max_ell}, '
              f'{len(all_unique)} unique diagrams, '
              f'{len(tau_grid)} '
              f'{"τ" if result.get("t_min") is None else "time"} points.')
        if phase_walls:
            total = sum(phase_walls.values())
            print(f'\n  Phase wall summary (Σ = {total:.1f}s):')
            for label, dt in phase_walls.items():
                pct = 100.0 * dt / total if total > 0 else 0.0
                print(f'    {label:12s}  {dt:6.2f}s  ({pct:4.1f}%)')

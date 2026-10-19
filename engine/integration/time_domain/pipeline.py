"""
engine.integration.time_domain.pipeline
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


from engine.integration.time_domain.final_integral import (
    integrate_diagram,
    PropagatorTD,
    _loop_number_from_graph,
)
from engine.errors import DaedalusError


# ───────────────────────────────────────────────────────────────────────
# Parallel-evaluation support (2026-04-21, parallel-eval branch)
# ───────────────────────────────────────────────────────────────────────
#
# ``compute_correction_td`` returns both a scalar ``total_C(*tau)`` (the
# serial, always-available entry point) and ``total_C_batch(tau_list,
# parallel=...)`` for fan-out over a τ grid.
#
# Parallelism uses ``multiprocessing.Pool`` with the ``fork`` start method,
# on Linux only.  Fork is chosen because the per-diagram ``contribution``
# callables returned by ``integrate_diagram`` are nested closures that
# standard ``pickle`` cannot serialise; with ``fork`` the workers inherit them
# from the parent's memory, and only the τ tuples and the complex outputs
# cross the process boundary.
#
# **macOS never forks** (``engine.fork_safety.fork_unsafe``): forking after
# Cocoa/BLAS/ECL initialisation has hard-crashed a Jupyter kernel and the OS
# of the dev machine, and a Sage process is multithreaded after its first
# symbolic step (Boehm GC markers), so there is no clean fork parent there.
# On macOS a parallel batch runs serially with one warning per process.  The
# package never sets ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY`` (the runtime
# reads it once at image load, so setting it from Python never affected these
# forks, and it leaked into every process exec'd later).  ``DAEDALUS_PARALLEL=0``
# turns every pool off.
#
# Windows has no ``fork`` start method; picklable state-space diagram tasks
# with ``spawn`` (macOS and Windows) are the planned follow-up.  Until then a
# Windows ``parallel=True`` batch raises ``ValueError: cannot find context for
# 'fork'``.
#
# ``_WORKER_STATE`` is a module-level dict that lets ``_worker_eval``
# (a picklable top-level function) reach the parent's ``total_C``
# callable after fork.  ``total_C_batch`` writes the current ``total_C``
# into ``_WORKER_STATE`` *before* creating the Pool; fork copies the
# dict into every worker, and each worker's ``_worker_eval`` looks it up
# from its own (inherited) module globals.  Because the Pool is
# created-and-destroyed per batch call, there is no stale-state risk
# across calls — subsequent calls overwrite the entry before forking a
# fresh set of workers.
#
# NOTE: the 2026-04-20 thread-pool attempt (see CHANGELOG) was reverted
# after a coincident ``_to_sr_ab`` precision regression confused the
# debugging — the parallelism itself was measured bit-identical at that
# time.  The current design: (a) uses process-based not thread-based
# parallelism, so there is no ECL/GIL contention; (b) pins bit-identity
# as a regression test so any drift is caught at the right layer; (c)
# keeps the serial ``total_C`` unchanged and purely additive, so a
# simple rollback of the batch API is always available.

_WORKER_STATE = {}

def _fork_unsafe(start_method):
    """Return True iff this batch must not fork: ``fork`` on macOS
    (:func:`engine.fork_safety.fork_unsafe`, one source of truth)."""
    from engine.fork_safety import fork_unsafe
    return fork_unsafe(start_method)


def _parallel_off():
    """``DAEDALUS_PARALLEL=0``: every batch serial, silently."""
    from engine.fork_safety import parallel_disabled
    return parallel_disabled()


def _warn_fork_guard_once():
    """The fork -> serial fallback warning, once per process."""
    from engine.fork_safety import warn_fork_fallback_once
    warn_fork_fallback_once('Phase J parallel batch')


def _worker_eval_total_C(tau_tuple):
    """Worker entry point for ``total_C_batch``'s fork-based Pool.

    Must be a module-level function so it can be pickled by
    ``multiprocessing``.  The actual ``total_C`` callable lives in
    ``_WORKER_STATE['total_C']`` and is inherited via fork (not via
    ``initargs`` pickling).
    """
    fn = _WORKER_STATE['total_C']
    return complex(fn(*tau_tuple))


def _worker_eval_one_diagram(task):
    """Worker entry point for the nested-parallel small-batch path:
    evaluate ONE diagram's contribution at ONE τ tuple.

    Task format: ``(tau_idx, diagram_idx, tau_tuple)``.

    The per-diagram ``contribution`` callables live in
    ``_WORKER_STATE['tree_callables']`` (inherited via fork).  Returns
    ``(tau_idx, diagram_idx, complex_value)`` so the parent can
    aggregate deterministically.
    """
    tau_idx, diagram_idx, tau_tuple = task
    fn = _WORKER_STATE['tree_callables'][diagram_idx]
    return (tau_idx, diagram_idx, complex(fn(*tau_tuple)))


def _ss_grid(points):
    """S3: the state-space route's τ batch over ``points`` (the raw tuples
    passed to ``total_C``): ``statespace.tau_grid_prefetch``, or a no-op
    context with ``DAEDALUS_PHASE_J_ENGINE=poles`` (the module is then not
    imported).  Only the serial evaluation paths use it (a forked worker
    would evaluate the whole grid for its share of it)."""
    import contextlib
    from engine import _env
    if _env.phase_j_engine() == 'poles':
        return contextlib.nullcontext()
    from engine.integration.time_domain import statespace as _ss
    return _ss.tau_grid_prefetch(points)


def _workers_within_budget(n_w):
    """The worker count a forked τ batch may use within the running Phase J
    memory budget (item 61): a worker builds its own copy of the lazily
    built stationary states and dirties the plans it reads, so a run that
    has charged ``used`` bytes holds up to ``(1 + n_w) x used`` with ``n_w``
    workers.  Outside a ``phase_j_memory_scope``, without a budget or with
    nothing charged: ``n_w`` unchanged.  ``<= 1``: run serially."""
    from engine.integration.time_domain import final_integral as _fi
    lock = getattr(_fi, '_PLAN_MEM_LOCK', None)
    if lock is None:
        return n_w
    with lock:
        if _fi._PLAN_MEM['active'] <= 0:
            return n_w
        budget, used = _fi._PLAN_MEM['budget'], _fi._PLAN_MEM['used']
    if budget is None or used <= 0:
        return n_w
    fit = int(budget // used) - 1
    if fit < n_w:
        _fi._RUNTIME_COUNTERS['parallel_workers_capped_by_budget'] = (
            _fi._RUNTIME_COUNTERS.get('parallel_workers_capped_by_budget', 0)
            + 1)
    return max(1, min(n_w, fit))


class DiscardedDiagramsError(DaedalusError, RuntimeError):
    """A callable of a Phase J run made with ``discard=True`` (item 61) was
    asked for a point it was not evaluated on, or for a diagram it
    released."""


def _discarded_callable(idx):
    def released(*_a, **_k):
        raise DiscardedDiagramsError(
            f'diagram #{idx} of this Phase J run was released after its '
            f'values on the requested points were taken (discard=True); '
            f'rerun compute_cumulants with discard=False to keep the '
            f'per-diagram callables.')
    released.discarded = True
    return released


def _mem_mark():
    """The running budget's charge counters now (``_mem_release``)."""
    from engine.integration.time_domain import final_integral as _fi
    with _fi._PLAN_MEM_LOCK:
        m = _fi._PLAN_MEM
        return (m['active'] > 0, m['used'], m['tuples'], m['plans'])


def _mem_release(mark, refund=True):
    """The bytes charged since ``mark`` (one diagram's plans and stationary
    states); with ``refund`` they are taken back from the running total
    (the diagram's closure has been dropped; item 61)."""
    from engine.integration.time_domain import final_integral as _fi
    scoped, used0, tuples0, plans0 = mark
    with _fi._PLAN_MEM_LOCK:
        m = _fi._PLAN_MEM
        if not scoped or m['active'] <= 0:
            return 0
        nb = m['used'] - used0
        if refund:
            m['used'], m['tuples'], m['plans'] = used0, tuples0, plans0
        return nb


def compute_correction_td(
    typed_diagrams=None,
    prefactors=None,
    propagator_data=None,
    k=None,
    num_params=None,
    ext_time_vars=None,
    origin_leaf_idx=0,
    external_fields=None,
    # Spatial loop integrator (Stage C.5): td -> per-edge EdgeModeSum builder.
    edge_mode_sums_builder_fn=None,
    # retired (audit W7, ACC-67): raises; pass typed_diagrams + prefactors
    kernel_groups=None,
    # M8: divergence policy of the box-free routes (``integrate_diagram``)
    on_divergence=None,
    # item 61: evaluate each diagram on these points right after it is
    # built, keep the values, release it (``discard``)
    discard_points=None,
):
    r"""
    Time-domain entry point: evaluate typed diagrams by Phase J vertex-time
    integration (closed forms on numeric poles; see
    ``final_integral.integrate_diagram``).

    Takes ``typed_diagrams`` + ``prefactors`` (one prefactor per diagram).
    The legacy ``kernel_groups=`` input is retired: it integrated every
    diagram of a group with the group's summed prefactor (audit ACC-67) and
    now raises ``TypeError``.

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
    discard_points : list of tuple or None
        Item 61, evaluate-and-discard: each diagram is evaluated on these
        points (``total_C``'s arguments, under the state-space τ batch)
        right after ``integrate_diagram`` builds it; the values are kept
        and the diagram's callable, plans and stationary states are
        released, their memory charge refunded to the running budget
        (``phase_j_memory_scope``), so the run holds one diagram at a
        time.  A diagram with shot-noise δ records keeps its charge (the
        records evaluate its subsets later).  ``total_C`` /
        ``total_C_batch`` / ``eval_per_diagram_batch`` then answer those
        points only (the same sums, in the same order) and raise
        ``DiscardedDiagramsError`` elsewhere; ``result['discard']`` has
        the bookkeeping.
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
    if hasattr(propagator_data, 'get') and propagator_data.get(
            'jordan_incomplete'):
        # a repeated pole: the stored residues miss the t exp(lambda t)
        # terms; compute_cumulants averages deformed propagators instead
        # (api._pole_clusters)
        raise ValueError(
            'propagator_data has a repeated pole (a Jordan block): its '
            'pole_vals / C_mats do not represent the propagator.  Use '
            'compute_cumulants, which resolves it (api._pole_clusters), or '
            'pass the propagator of a deformation node.')
    if on_divergence is not None:
        from engine.integration.time_domain.final_integral import (
            _divergence_policy)
        _divergence_policy(on_divergence)               # validate early
    from engine.integration.time_domain.final_integral import (
        _warn_retired_env)
    _warn_retired_env()
    if kernel_groups is not None:
        raise TypeError(
            'compute_correction_td(kernel_groups=...) is retired: pass '
            'typed_diagrams= and prefactors= (one prefactor per diagram)')
    if typed_diagrams is None:
        typed_diagrams = []
        prefactors = []

    if ext_time_vars is None:
        ext_time_vars = [
            SR.var(f't_{j+1}', latex_name=rf't_{{{j+1}}}')
            for j in range(k)
        ]

    # M5 L2: the model-level data every diagram of THIS call shares (the
    # G(t) matrix, the poles and residues); it dies with the call, so the
    # spatial bridge's per-q re-solve of the poles cannot be served stale.
    prop_td = (PropagatorTD(propagator_data, num_params)
               if (typed_diagrams and hasattr(propagator_data, 'get'))
               else None)

    tree_callables = []
    groups_out = []
    skipped = []
    all_delta_contributions = []
    all_delta_contributions_multi = []
    n_delta_subsets_dropped = 0

    if discard_points is not None:
        discard_points = [tuple(pt) for pt in discard_points]
    discard_vals = []           # per kept diagram: complex values per point
    discard_info = {'points': discard_points, 'diagrams': 0, 'refunded': 0,
                    'kept_for_delta': 0, 'built_bytes': 0,
                    'max_diagram_bytes': 0}

    for idx, (td, pf) in enumerate(zip(typed_diagrams, prefactors)):
        loop_number = _loop_number_from_graph(td)
        mark = _mem_mark() if discard_points is not None else None

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
            edge_mode_sums_builder=(
                edge_mode_sums_builder_fn(td)
                if edge_mode_sums_builder_fn is not None else None),
            prop_td=prop_td,
            # only when set: an A/B harness may run an older integrate_diagram
            **({} if on_divergence is None
               else {'on_divergence': on_divergence}),
        )
        if result['status'] == 'ok':
            contribution = result['contribution']
            if discard_points is not None:
                with _ss_grid(discard_points):
                    discard_vals.append([complex(contribution(*pt))
                                         for pt in discard_points])
                contribution = _discarded_callable(idx)
                result['contribution'] = contribution
                has_delta = bool(result.get('delta_contributions')
                                 or result.get('delta_contributions_multi'))
                nb = _mem_release(mark, refund=not has_delta)
                discard_info['diagrams'] += 1
                discard_info['built_bytes'] += nb
                discard_info['max_diagram_bytes'] = max(
                    discard_info['max_diagram_bytes'], nb)
                if has_delta:
                    discard_info['kept_for_delta'] += 1
                else:
                    discard_info['refunded'] += 1
            tree_callables.append(contribution)
            n_delta_subsets_dropped += result.get(
                'n_delta_subsets_dropped', 0)
            for _dc in result.get('delta_contributions', []):
                # Feature B provenance: index into ``typed_diagrams``
                _dc['diagram_index'] = idx
                all_delta_contributions.append(_dc)
            for _dc in result.get('delta_contributions_multi', []):
                _dc['diagram_index'] = idx
                all_delta_contributions_multi.append(_dc)
            # Per-subset evaluator labels (e.g.
            # 'polygon_modesum', 'fast_numpy', 'fast_callable') —
            # bubble these up so the caller can see at a glance
            # whether the analytic m=2 polygon integrator is firing.
            subset_diags = result.get('subset_diagnostics', []) or []
            subset_evaluators = [
                d.get('evaluator') for d in subset_diags
                if isinstance(d, dict) and d.get('status') == 'evaluated'
            ]
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
                'subset_evaluators': subset_evaluators,
                'subset_m_values': [
                    d.get('m_after_delta')
                    for d in subset_diags
                    if isinstance(d, dict) and d.get('status') == 'evaluated'
                ],
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

    if discard_points is not None:
        # the same sums as below, from the kept values, in diagram order
        _at = {}
        for p_idx, pt in enumerate(discard_points):
            tot = 0.0 + 0.0j
            for vals in discard_vals:
                tot = tot + complex(vals[p_idx])
            _at.setdefault(pt, (p_idx, tot))

        def _lookup(pt):
            hit = _at.get(tuple(pt))
            if hit is None:
                raise DiscardedDiagramsError(
                    f'this Phase J run used discard=True: its diagrams were '
                    f'evaluated on the requested points only and released; '
                    f'{tuple(pt)} is not one of them.  Rerun '
                    f'compute_cumulants with discard=False to evaluate '
                    f'elsewhere.')
            return hit

        def total_C(*ext_time_values):
            if not tree_callables:
                return 0.0 + 0.0j
            return _lookup(ext_time_values)[1]

        def total_C_batch(tau_points, parallel=True, n_workers=None,
                          start_method='fork'):
            return [total_C(*pt) for pt in tau_points]

        def eval_per_diagram_batch(tau_points, parallel=True, n_workers=None,
                                   start_method='fork'):
            loop_nums = [g['loop_number'] for g in groups_out
                         if g.get('contribution') is not None]
            return ([[vals[_lookup(pt)[0]] for vals in discard_vals]
                     for pt in tau_points], loop_nums)

        return _td_result(total_C, total_C_batch, eval_per_diagram_batch,
                          all_delta_contributions, n_delta_subsets_dropped,
                          all_delta_contributions_multi, groups_out, skipped,
                          ext_time_vars, discard_info)

    def total_C(*ext_time_values):
        if not tree_callables:
            return 0.0 + 0.0j
        total = 0.0 + 0.0j
        for fn in tree_callables:
            val = fn(*ext_time_values)
            total = total + complex(val)
        return total

    def total_C_batch(tau_points, parallel=True, n_workers=None,
                      start_method='fork'):
        """Evaluate ``total_C`` on a list of τ points, optionally in
        parallel across processes.

        Two parallelism strategies, auto-selected by batch shape:

        * **Per-τ parallelism** — each worker evaluates the FULL
          ``total_C(τ_i)`` (summing all diagrams serially inside the
          worker).  Used when ``len(tau_points) >= n_workers`` — the
          simpler, cheaper coordination path.  Matches the 2026-04-21
          initial design.

        * **Per-(τ, diagram) nested parallelism** — each worker
          evaluates ONE diagram's contribution at ONE τ tuple; the
          parent aggregates.  Used when ``len(tau_points) <
          n_workers`` so even a single-τ call can still use many cores.
          Essential for ``SIM_MODE='point'`` where the batch is 1-4 τ
          points — the old code ran serial on those, wasting all
          available parallelism.

        The decision is made once per call based on total task count.

        Parameters
        ----------
        tau_points : iterable of tuples
            Each element is a k-tuple passed through as
            ``total_C(*tau_tuple)``.  Order is preserved in the output.
        parallel : bool, default True
            If True, fan the evaluations out across worker processes.
            If False, run serial (equivalent to
            ``[total_C(*pt) for pt in tau_points]``).
        n_workers : int or None, default None
            Worker process count.  ``None`` picks
            ``min(os.cpu_count(), total_tasks)``.
        start_method : {'fork', 'spawn', 'forkserver'}, default 'fork'
            Multiprocessing start method.  ``'fork'`` is required for
            the current design because the per-diagram ``contribution``
            closures are unpicklable (nested functions); fork inherits
            them via process memory.  ``'spawn'`` and ``'forkserver'``
            will raise a pickling error in the current implementation.
            On macOS ``'fork'`` is never used: the batch runs serially
            with one warning per process.

        Returns
        -------
        list of complex
            ``total_C`` evaluated at each input point, in input order.

        Notes
        -----
        ``DAEDALUS_PARALLEL=0`` makes the batch serial.  Bit-identity with
        the serial path is pinned by
        ``test_phase_J_total_C_batch_parallel_matches_serial`` (per-τ)
        and ``test_phase_J_total_C_batch_nested_matches_serial``
        (nested).
        """
        tau_list = [tuple(pt) for pt in tau_points]
        # Kill switch, then the fork guard (see module header): macOS never
        # forks; the batch runs serially there.
        if parallel and _parallel_off():
            parallel = False
        if parallel and _fork_unsafe(start_method):
            _warn_fork_guard_once()
            parallel = False
        if not parallel or len(tau_list) == 0:
            with _ss_grid(tau_list):
                return [total_C(*pt) for pt in tau_list]

        import multiprocessing as mp
        import os

        n_diag = len(tree_callables)
        if n_diag == 0:
            return [0.0 + 0.0j] * len(tau_list)

        n_w_cap = _workers_within_budget(
            n_workers if n_workers is not None else (os.cpu_count() or 4))
        total_tasks = len(tau_list) * n_diag

        # Serial fast path: not enough work to amortise pool setup.
        # A fork+join costs ~50-200 ms on macOS/Linux; require at least
        # a handful of tasks per worker before it pays off.
        if n_w_cap <= 1 or total_tasks < max(4, 2 * n_w_cap):
            with _ss_grid(tau_list):
                return [total_C(*pt) for pt in tau_list]

        ctx = mp.get_context(start_method)

        if len(tau_list) >= n_w_cap:
            # ── Per-τ parallel path (original design) ────────────────
            # Enough τ points to fill the worker pool.  Each worker
            # handles its share of τ's and sums all diagrams serially
            # within the worker.  Simpler coordination, same bit-
            # identity as serial.
            _WORKER_STATE['total_C'] = total_C
            n_w = min(n_w_cap, len(tau_list))
            with ctx.Pool(processes=n_w) as pool:
                return pool.map(_worker_eval_total_C, tau_list)

        # ── Per-(τ, diagram) nested parallel path (small-batch) ──
        # Fewer τ points than workers.  Spread ALL (τ, diagram) pairs
        # across workers and aggregate in the parent.  This is the
        # right strategy for SIM_MODE='point' (1-4 τ points).
        _WORKER_STATE['tree_callables'] = tree_callables
        tasks = [(p_idx, d_idx, pt)
                 for p_idx, pt in enumerate(tau_list)
                 for d_idx in range(n_diag)]
        n_w = min(n_w_cap, len(tasks))
        with ctx.Pool(processes=n_w) as pool:
            partial = pool.map(_worker_eval_one_diagram, tasks)
        # Aggregate into (n_tau,) complex array.  Sum per τ in the
        # SAME diagram order as serial ``total_C`` (ascending d_idx),
        # so floating-point accumulation is bit-identical.
        out = [0.0 + 0.0j] * len(tau_list)
        buckets = [[None] * n_diag for _ in tau_list]
        for (p_idx, d_idx, val) in partial:
            buckets[p_idx][d_idx] = val
        for p_idx in range(len(tau_list)):
            s = 0.0 + 0.0j
            for d_idx in range(n_diag):
                s = s + buckets[p_idx][d_idx]
            out[p_idx] = s
        return out

    def eval_per_diagram_batch(tau_points, parallel=True, n_workers=None,
                                start_method='fork'):
        """Evaluate EACH diagram's contribution at EACH τ point, in
        parallel, returning the full (n_tau, n_diagram) grid.

        Useful when downstream code needs BOTH the per-diagram breakdown
        AND the total: call this once, then aggregate in the caller.
        Without this helper, cell 31-style workflows would run scipy
        serially for every (τ, diagram) pair — prohibitive at k=2
        ell=1 (100+ diagrams) and painful at k=3.

        Returns
        -------
        vals : list of list of complex
            ``vals[tau_idx][diag_idx]`` is the diagram-level contribution
            at ``tau_points[tau_idx]``.  The columns are the EVALUATED
            diagrams in input order: a skipped diagram has no column, so
            column ``j`` is the ``j``-th entry of ``'groups'`` that carries
            a ``'contribution'`` (its ``'kernel_id'`` is the index into
            ``typed_diagrams``), not ``typed_diagrams[j]`` in general.
        diag_loop_numbers : list of int
            ``loop_number`` per diagram, same order as the inner lists.
        """
        tau_list = [tuple(pt) for pt in tau_points]
        n_diag = len(tree_callables)
        loop_nums = [g.get('loop_number', 0) for g in groups_out
                     if g.get('contribution') is not None]
        if n_diag == 0 or len(tau_list) == 0:
            return [[] for _ in tau_list], loop_nums

        total_tasks = len(tau_list) * n_diag
        n_w_cap = _workers_within_budget(
            n_workers if n_workers is not None
            else (__import__('os').cpu_count() or 4))

        # Kill switch, then the fork guard (see module header).
        if parallel and _parallel_off():
            parallel = False
        if parallel and _fork_unsafe(start_method):
            _warn_fork_guard_once()
            parallel = False

        if (not parallel) or n_w_cap <= 1 or total_tasks < max(
                4, 2 * n_w_cap):
            # Serial path.  Still honours original diagram order.
            vals = []
            with _ss_grid(tau_list):
                for pt in tau_list:
                    row = []
                    for fn in tree_callables:
                        row.append(complex(fn(*pt)))
                    vals.append(row)
            return vals, loop_nums

        import multiprocessing as mp
        _WORKER_STATE['tree_callables'] = tree_callables
        tasks = [(p_idx, d_idx, pt)
                 for p_idx, pt in enumerate(tau_list)
                 for d_idx in range(n_diag)]
        ctx = mp.get_context(start_method)
        n_w = min(n_w_cap, len(tasks))
        with ctx.Pool(processes=n_w) as pool:
            partial = pool.map(_worker_eval_one_diagram, tasks)
        vals = [[0.0 + 0.0j] * n_diag for _ in tau_list]
        for (p_idx, d_idx, val) in partial:
            vals[p_idx][d_idx] = val
        return vals, loop_nums

    return _td_result(total_C, total_C_batch, eval_per_diagram_batch,
                      all_delta_contributions, n_delta_subsets_dropped,
                      all_delta_contributions_multi, groups_out, skipped,
                      ext_time_vars, None)


def _td_result(total_C, total_C_batch, eval_per_diagram_batch,
               all_delta_contributions, n_delta_subsets_dropped,
               all_delta_contributions_multi, groups_out, skipped,
               ext_time_vars, discard_info):
    out = {
        'total_C': total_C,
        'total_C_batch': total_C_batch,
        'eval_per_diagram_batch': eval_per_diagram_batch,
        'delta_contributions': all_delta_contributions,
        # Feature B: shot-noise subsets that gave no record (see
        # ``integrate_diagram``); a failed diagram is in ``skipped_kernel_ids``
        'n_delta_subsets_dropped': n_delta_subsets_dropped,
        # δ spikes of codimension > 1 (several external-time equalities)
        'delta_contributions_multi': all_delta_contributions_multi,
        'groups': groups_out,
        'skipped_kernel_ids': skipped,
        'ext_time_vars': ext_time_vars,
    }
    if discard_info is not None:
        out['discard'] = discard_info
    return out

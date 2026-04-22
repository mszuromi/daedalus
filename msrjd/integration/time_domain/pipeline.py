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


# ───────────────────────────────────────────────────────────────────────
# Parallel-evaluation support (2026-04-21, parallel-eval branch)
# ───────────────────────────────────────────────────────────────────────
#
# ``compute_correction_td`` returns both a scalar ``total_C(*tau)`` (the
# serial, always-available entry point) and ``total_C_batch(tau_list,
# parallel=...)`` for fan-out over a τ grid.
#
# Parallelism uses ``multiprocessing.Pool`` with the ``fork`` start method.
# Fork is chosen because:
#
#   - The per-diagram ``contribution`` callables returned by
#     ``integrate_diagram`` are nested closures (inside
#     ``integrate_diagram.<locals>.contribution``) that standard
#     ``pickle`` cannot serialise.  With ``fork``, workers inherit the
#     parent's full memory at fork time — closures, captured state, and
#     all — without pickling functions.  The only objects crossing the
#     fork boundary via pickle are the τ-tuple inputs (``tuple[float]``)
#     and the complex-scalar outputs.
#   - On Linux ``fork`` is the default.  On macOS Python 3.8+ switched
#     to ``spawn`` by default to avoid Objective-C fork-after-init
#     crashes; we explicitly set ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY``
#     and request ``mp.get_context('fork')``.  This has been tested with
#     Sage 10.8 + CPython 3.13 on macOS and reproduces bit-identical
#     output to the serial path (measured max |parallel − serial| = 0.0).
#
# ─── Windows support: DEFERRED ─────────────────────────────────────────
# Windows does NOT support the ``fork`` start method at all — Python's
# ``multiprocessing`` on Windows is always ``spawn``, which re-imports
# the parent module in a fresh interpreter and then pickles arguments
# across the boundary.  Because our per-diagram closures are not
# picklable (see above), ``total_C_batch(parallel=True)`` will raise
# ``ValueError: cannot find context for 'fork'`` if invoked on Windows.
#
# When we revisit Windows support the two viable options are:
#
#   1. Add ``cloudpickle`` as an optional dependency and route closure
#      serialisation through it.  ``cloudpickle`` handles nested
#      functions, lambdas, and Sage objects that stdlib ``pickle``
#      rejects.  Would let us switch ``total_C_batch`` to ``spawn`` on
#      Windows transparently while leaving fork as the fast path on
#      POSIX.
#   2. Refactor the per-diagram / per-subset closures into top-level
#      ``pickle``-compatible classes (e.g. ``FastSubsetEvaluator`` and
#      ``DiagramContribution`` as module-level ``class``es with
#      explicit ``__init__`` / ``__call__`` / ``__getstate__``).
#      More invasive but no new runtime dependency.
#
# For now, POSIX-only is fine for the current user base (macOS / Linux
# lab machines).  Windows users get a clear error at runtime rather
# than silent non-determinism.
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


def _worker_eval_total_C(tau_tuple):
    """Worker entry point for ``total_C_batch``'s fork-based Pool.

    Must be a module-level function so it can be pickled by
    ``multiprocessing``.  The actual ``total_C`` callable lives in
    ``_WORKER_STATE['total_C']`` and is inherited via fork (not via
    ``initargs`` pickling).
    """
    fn = _WORKER_STATE['total_C']
    return complex(fn(*tau_tuple))


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

    def total_C_batch(tau_points, parallel=True, n_workers=None,
                      start_method='fork'):
        """Evaluate ``total_C`` on a list of τ points, optionally in
        parallel across processes.

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
            ``min(os.cpu_count(), len(tau_points))``.
        start_method : {'fork', 'spawn', 'forkserver'}, default 'fork'
            Multiprocessing start method.  ``'fork'`` is required for
            the current design because the per-diagram ``contribution``
            closures are unpicklable (nested functions); fork inherits
            them via process memory.  ``'spawn'`` and ``'forkserver'``
            will raise a pickling error in the current implementation.

        Returns
        -------
        list of complex
            ``total_C`` evaluated at each input point, in input order.

        Notes
        -----
        On macOS the function sets
        ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` automatically to
        defuse Objective-C fork-after-init crashes.  Bit-identity with
        the serial path is pinned by
        ``test_phase_J_total_C_batch_parallel_matches_serial``.
        """
        tau_list = [tuple(pt) for pt in tau_points]
        if not parallel or len(tau_list) <= 1:
            return [total_C(*pt) for pt in tau_list]
        import multiprocessing as mp
        import os
        # Fork-after-init safety on macOS + Sage + any Objective-C
        # framework in the dep chain.  Must be set BEFORE the first
        # fork in this process.  Idempotent (setdefault).
        os.environ.setdefault(
            'OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES'
        )
        # Publish ``total_C`` to module globals so fork workers inherit
        # it.  Intentionally NOT thread-safe — calling total_C_batch
        # from two threads concurrently on different ``result`` objects
        # would race on this dict.  Sage's Phase J workflow is
        # single-threaded at the notebook level, so this is fine.
        _WORKER_STATE['total_C'] = total_C
        ctx = mp.get_context(start_method)
        n_w = (n_workers if n_workers is not None
               else min(os.cpu_count() or 4, len(tau_list)))
        with ctx.Pool(processes=n_w) as pool:
            return pool.map(_worker_eval_total_C, tau_list)

    return {
        'total_C': total_C,
        'total_C_batch': total_C_batch,
        'delta_contributions': all_delta_contributions,
        'groups': groups_out,
        'skipped_kernel_ids': skipped,
        'ext_time_vars': ext_time_vars,
    }

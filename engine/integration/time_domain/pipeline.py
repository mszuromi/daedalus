"""
engine.integration.time_domain.pipeline
======================================
Time-domain correlator evaluation (tree and loop diagrams).

Evaluates each typed diagram directly in the time domain, vertex time
integrals done in closed form (state-space route or pole-residue modesum;
numerical quadrature only as a fallback).  No frequency-domain integral
construction or loop-kernel grouping is needed — the pipeline works
directly from:

  1. A list of typed diagrams (from the enumeration pipeline)
  2. Their scalar prefactors (from ``classify_coefficient_factors``)
  3. The retarded propagator in pole-residue form (``pole_vals``,
     ``C_mats``, ``D_delta``)

Every diagram, tree (loop_number == 0) or loop (loop_number > 0), is
evaluated independently by ``integrate_diagram`` and the contributions are
summed per loop order; a diagram that cannot be evaluated is reported in the
result (``status = 'incomplete'``), never dropped silently.

Convention
----------
``total_C(t_1, t_2, ..., t_k)`` returns the sum of all diagram
contributions.  Position i is ALWAYS the time of
``external_fields[i]``:

  - ``external_fields[0]`` → ``t_1`` (base time, pinned to 0)
  - ``external_fields[1]`` → ``t_2``, ``τ_1 = t_2 - t_1``
  - ``external_fields[2]`` → ``t_3``, ``τ_2 = t_3 - t_1``

This is enforced by the canonical time remapping in
``integrate_diagram``.
"""

from sage.all import SR


from engine.integration.time_domain.final_integral import (
    integrate_diagram,
    PropagatorTD,
    _loop_number_from_graph,
)
from engine.errors import DaedalusError


# ───────────────────────────────────────────────────────────────────────
# Parallel evaluation of a τ batch
# ───────────────────────────────────────────────────────────────────────
#
# ``compute_correction_td`` returns a scalar ``total_C(*tau)`` (the serial,
# always-available entry point) and ``total_C_batch(tau_list, parallel=...)``
# / ``eval_per_diagram_batch`` for a τ grid.
#
# **Partition by diagram only.**  A worker evaluates WHOLE diagrams on the
# WHOLE grid, inside its own ``_ss_grid`` (the state-space prefetch steps the
# last segment from one grid length to the next, so splitting the grid would
# change rounding), and the parent sums the diagram values of each point in
# diagram order, exactly as ``total_C`` does.  The result is bit-identical
# to the serial batch for any worker count and either method:
#
# * **threads** (every platform, notebooks included: nothing forks).  Only a
#   diagram whose largest stationary state block has at least
#   ``THREAD_BLOCK_MIN`` entries goes to the pool: its Kronecker / BLAS work
#   releases the GIL (measured 2.7-3.0x at 4 threads), while small diagrams
#   are dispatch-bound and slower on threads (0.75-0.86x measured), so the
#   main thread evaluates those meanwhile.  Largest blocks first.
# * **fork** (Linux only; ``engine.fork_safety.fork_unsafe``).  The workers
#   inherit the per-diagram closures (nested functions that ``pickle``
#   cannot serialise) through ``_FANOUT``; each returns its diagrams' values
#   and its ``_RUNTIME_COUNTERS`` delta.  Used when the batch has enough
#   diagrams to amortise the pool, capped by the memory budget (per-worker
#   budget shares).
#
# **macOS never forks** (a fork after Cocoa/BLAS/ECL initialisation has
# hard-crashed a Jupyter kernel and the OS; a Sage process is multithreaded
# after its first symbolic step), and the package never sets
# ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY``.  ``DAEDALUS_PARALLEL=0`` turns
# every pool off.  Worker threads collect their counter increments in a
# thread-local overlay (``final_integral.counter_delta``) that the parent
# merges in diagram order, so no increment is lost to a race.

#: Smallest largest-stationary-block size (entries) of a diagram's
#: state-space subsets for which a worker thread pays (calibration:
#: blocks >= 256 gave 2.7x at 4 threads, blocks <= 81 ran 0.75-0.86x).
THREAD_BLOCK_MIN = 256
#: Default thread count of the fan-out (``n_workers`` overrides).
DEFAULT_THREADS = 8
#: Automatic spawned workers (macOS / Windows, outside a Jupyter kernel):
#: only when a timing sample of the batch predicts at least this many
#: seconds of serial work (a worker starts in ~0.3 s; measured).
SPAWN_MIN_SECONDS = 3.0
#: Default spawned worker count (``n_workers`` overrides).
DEFAULT_SPAWN_WORKERS = 4
#: Opt-in for automatic spawned workers inside a Jupyter kernel (explicit
#: ``start_method='spawn'`` is the other opt-in).  Off until the manual
#: notebook checklist has passed on the user's machine.
SPAWN_KERNEL_ENV = 'DAEDALUS_SPAWN_IN_KERNEL'

#: Fork workers' view of the batch (set before the pool forks).
_FANOUT = {}
#: Kept for ``api.compute._counter_snapshot`` (identity check); the fork
#: workers now return their counters, so it no longer changes.
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


def _diagram_block(fn):
    """The largest stationary-state block (entries) over the state-space
    subsets of one diagram's contribution (0: none on the route)."""
    parts = getattr(fn, 'phase_j_parts', None) or {}
    b = 0
    for s in parts.get('subsets', ()):
        sub = getattr(s, 'ss_subset', None)
        if sub is not None and not getattr(sub, 'empty', False):
            b = max(b, sub.max_width()[1])
    return b


def _eval_diagram(fn, tau_list):
    """One diagram on the whole grid, in its own prefetch context."""
    with _ss_grid(tau_list):
        return [complex(fn(*pt)) for pt in tau_list]


def _thread_task(fn, tau_list):
    from engine.integration.time_domain import final_integral as _fi
    with _fi.counter_delta() as cd:
        vals = _eval_diagram(fn, tau_list)
    return vals, cd.delta


def _fork_task(chunk):
    """Fork worker: the diagrams ``chunk`` (indices) on the batch's grid."""
    from engine.integration.time_domain import final_integral as _fi
    fns, tau_list = _FANOUT['fns'], _FANOUT['tau_list']
    with _fi.counter_delta() as cd:
        out = [(d, _eval_diagram(fns[d], tau_list)) for d in chunk]
    return out, cd.delta


def _switch_on_now():
    from engine import _env
    if _env.phase_j_engine() == 'poles':
        return False
    from engine.integration.time_domain import statespace as _ss
    return _ss.switch_on_active()


def _fan_out(fns, tau_list, parallel, n_workers, start_method):
    """``vals[d][p]`` (complex) of every diagram ``d`` at every point ``p``,
    computed with a worker pool, or ``None`` when the batch should run
    serially (no parallelism asked or allowed, or nothing a pool would
    speed up).  See the module notes above for the methods."""
    n = len(fns)
    if not parallel or n == 0 or not tau_list or _parallel_off():
        return None
    if start_method not in (None, 'fork', 'threads', 'spawn'):
        raise ValueError(
            f'start_method={start_method!r}: a Phase J batch runs on threads, '
            f"spawned workers or (Linux) fork; expected None, 'threads', "
            f"'spawn' or 'fork'.")
    if _switch_on_now():
        return None             # a switch-on scope is per thread: stay here
    import os
    want = int(n_workers) if n_workers is not None else None
    fork_ok = start_method in (None, 'fork') and not _fork_unsafe('fork')
    if start_method == 'fork' and not fork_ok:
        _warn_fork_guard_once()
    if fork_ok:
        n_w = _workers_within_budget(want or (os.cpu_count() or 4))
        if n_w > 1 and n >= max(4, 2 * n_w):
            return _fork_fan_out(fns, tau_list, n_w)
    vals = [None] * n
    if _spawn_allowed(start_method, fork_ok):
        done = _spawn_fan_out(fns, tau_list, want or DEFAULT_SPAWN_WORKERS,
                              explicit=start_method == 'spawn', vals=vals)
        if done:
            return vals
    # threads; diagrams a declined spawn sample already evaluated keep
    # their values
    blocks = [_diagram_block(fn) for fn in fns]
    heavy = [d for d in range(n)
             if blocks[d] >= THREAD_BLOCK_MIN and vals[d] is None]
    n_t = min(want or DEFAULT_THREADS, os.cpu_count() or 4, len(heavy))
    if n_t >= 2:
        # a thread's own memory is its segment work (the states are
        # shared and already charged): a few vectors of the largest block
        from engine import worker_budget as _wb
        n_t, _share = _wb.workers_that_fit(
            n_t, 2**20 + 64 * max(blocks[d] for d in heavy),
            'parallel_workers_capped_by_budget')
    if n_t < 2:
        if all(v is None for v in vals):
            return None
        _fill_serial(fns, tau_list, vals)
        return vals
    return _thread_fan_out(fns, tau_list, heavy, blocks, n_t, vals)


def _fill_serial(fns, tau_list, vals, which=None):
    """Evaluate here, in one prefetch context, every diagram of ``which``
    (default: every one) whose ``vals`` entry is still ``None``."""
    with _ss_grid(tau_list):
        for d in (range(len(fns)) if which is None else which):
            if vals[d] is None:
                vals[d] = [complex(fns[d](*pt)) for pt in tau_list]


def _spawn_allowed(start_method, fork_ok):
    """Spawned workers: asked for explicitly, or automatically where fork
    is not available (macOS, Windows) outside a Jupyter kernel (inside one
    only with ``DAEDALUS_SPAWN_IN_KERNEL=1``)."""
    if start_method == 'spawn':
        return True
    if start_method is not None or fork_ok:
        return False
    from engine import _env
    from engine.fork_safety import in_jupyter_kernel
    return (not in_jupyter_kernel()) or _env.flag(SPAWN_KERNEL_ENV, False)


def _spawn_tasks(order, payloads, costs, n_w):
    """Chunks of ``(index, payload)`` (largest first), about
    ``1 / (8 n_w)`` of the total cost each, so small diagrams travel
    together and the large ones balance."""
    total = sum(costs[d] for d in order) or 1
    target = total / (8 * n_w)
    tasks, cur, acc = [], [], 0
    for d in order:
        cur.append((d, payloads[d]))
        acc += costs[d]
        if acc >= target or len(cur) >= 256:
            tasks.append(cur)
            cur, acc = [], 0
    if cur:
        tasks.append(cur)
    return tasks


def _plain_times(tau_list):
    """Whether every external time is a plain number (a spawned worker gets
    exactly these values; a symbolic time would make it import Sage)."""
    import numbers
    return all(isinstance(t, numbers.Real) and not isinstance(t, bool)
               and type(t).__module__ in ('builtins', 'numpy')
               for pt in tau_list for t in pt)


def _spawn_settings():
    """What a spawned worker takes over from this process: the running
    Phase J budget (its segment checks then decide as here) and the module
    settings of ``statespace`` / ``_light``."""
    from engine import _spawn_worker as _sw
    from engine.integration.time_domain import _light
    from engine.integration.time_domain import statespace as _ss
    return {'budget': tuple(_ss.budget_view()),
            'statespace': _sw.module_settings(_ss),
            'light': _sw.module_settings(_light)}


def _spawn_fan_out(fns, tau_list, n_w, explicit, vals):
    """Spawned-worker fan-out (``engine.spawn_pool``) into ``vals``: every
    diagram whose numeric task can leave this process
    (``statespace.diagram_payload``) goes to the pool; the rest, and any
    diagram a worker could not evaluate, are evaluated here.  Automatic mode
    first times a sample of diagrams here (their values are final) and
    spawns only when the rest is predicted to take at least
    ``SPAWN_MIN_SECONDS``.  Returns True when ``vals`` is complete; False:
    nothing more was done (the caller goes on with threads or serially).
    Explicit ``'spawn'`` raises when the pool cannot start; automatic mode
    falls back."""
    import threading
    import time
    from engine import spawn_pool as _sp
    from engine import worker_budget as _wb
    from engine.integration.time_domain import final_integral as _fi
    from engine.integration.time_domain import statespace as _ss
    n = len(fns)
    if not _plain_times(tau_list):
        return False
    payloads = [_ss.diagram_payload(fn) for fn in fns]
    elig = [d for d in range(n) if payloads[d] is not None]
    n_w = min(int(n_w), len(elig))
    if n_w < 2 or len(elig) < 2 * n_w:
        return False
    costs = {d: (_diagram_block(fns[d]) + 1) * len(payloads[d]['subsets'])
             for d in elig}
    if not explicit:
        step = max(1, len(elig) // 8)
        sample = elig[::step][:8]
        t0 = time.perf_counter()
        for d in sample:
            vals[d] = _eval_diagram(fns[d], tau_list)
        per = (time.perf_counter() - t0) / len(sample)
        elig = [d for d in elig if vals[d] is None]
        if per * len(elig) < SPAWN_MIN_SECONDS or len(elig) < 2 * n_w:
            return False
    order = sorted(elig, key=lambda d: (-costs[d], d))
    tasks = _spawn_tasks(order, payloads, costs, n_w)
    # a worker holds one task's states, programs and work at a time
    task_bytes = max(sum(sub.state_bytes() for i, pl in t
                         for sub, _p, _e in pl['subsets']) for t in tasks)
    n_fit, share = _wb.workers_that_fit(
        n_w, _wb.SPAWN_WORKER_BASE + 3 * task_bytes,
        'parallel_workers_capped_by_budget')
    if n_fit < 2:
        return False
    if n_fit < n_w:
        n_w = n_fit
        tasks = _spawn_tasks(order, payloads, costs, n_w)
    shared = _ss.payload_shared([payloads[d] for d in elig])
    try:
        pool = _sp.SpawnPool(n_w, shared, _spawn_settings())
    except Exception:
        if explicit:
            raise
        _fi._RUNTIME_COUNTERS['parallel_spawn_start_failed'] = (
            _fi._RUNTIME_COUNTERS.get('parallel_spawn_start_failed', 0) + 1)
        return False
    elig_set = set(elig)
    box = {}
    try:
        blobs = pool.encode(tasks)

        def run():
            try:
                box['out'] = pool.map(tasks, tau_list, blobs=blobs)
            except BaseException as exc:            # pragma: no cover
                box['err'] = exc
        th = threading.Thread(target=run, daemon=True,
                              name='daedalus-spawn-map')
        with _wb.FootprintWatch(pool.pids, share, _wb.kill_pid):
            th.start()
            # the diagrams that stay here, meanwhile
            _fill_serial(fns, tau_list, vals,
                         [d for d in range(n) if d not in elig_set])
            th.join()
    finally:
        pool.close()
    if 'err' in box:
        raise box['err']
    got, states, counters, lost = box['out']
    for d, v in got.items():
        vals[d] = v
    for d, st in states.items():       # keep the states built in the workers
        subs = payloads[d]['subsets']
        for j, stat, rho in st:
            sub = subs[j][0]
            with sub._lock:
                if sub._stat is None:
                    sub._stat = stat
                    sub.rho = max(sub.rho, rho)
    for delta in counters:
        _fi.merge_counter_delta(delta)
    if pool.sage_loaded:                            # never expected
        _fi._RUNTIME_COUNTERS['parallel_spawn_sage_loaded'] = (
            _fi._RUNTIME_COUNTERS.get('parallel_spawn_sage_loaded', 0) + 1)
    if lost:
        _fi._RUNTIME_COUNTERS['parallel_spawn_diagrams_here'] = (
            _fi._RUNTIME_COUNTERS.get('parallel_spawn_diagrams_here', 0)
            + len(lost))
        _fill_serial(fns, tau_list, vals, lost)
    _fi._RUNTIME_COUNTERS['parallel_spawn_batches'] = (
        _fi._RUNTIME_COUNTERS.get('parallel_spawn_batches', 0) + 1)
    return True


def _thread_fan_out(fns, tau_list, heavy, blocks, n_t, vals):
    from concurrent.futures import ThreadPoolExecutor
    from engine.integration.time_domain import final_integral as _fi
    heavy_set = set(heavy)
    ex = ThreadPoolExecutor(max_workers=n_t,
                            thread_name_prefix='daedalus-phase-j')
    try:
        futs = {d: ex.submit(_thread_task, fns[d], tau_list)
                for d in sorted(heavy, key=lambda d: (-blocks[d], d))}
        # the small diagrams, meanwhile (here)
        _fill_serial(fns, tau_list, vals,
                     [d for d in range(len(fns)) if d not in heavy_set])
        for d in sorted(heavy):
            vals[d], delta = futs[d].result()
            _fi.merge_counter_delta(delta)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    _fi._RUNTIME_COUNTERS['parallel_thread_batches'] = (
        _fi._RUNTIME_COUNTERS.get('parallel_thread_batches', 0) + 1)
    return vals


def _fork_fan_out(fns, tau_list, n_w):
    import multiprocessing as mp
    from engine.integration.time_domain import final_integral as _fi
    n = len(fns)
    size = max(1, -(-n // (4 * n_w)))
    chunks = [list(range(a, min(n, a + size))) for a in range(0, n, size)]
    from engine import worker_budget as _wb
    n_fit, share = _wb.workers_that_fit(n_w, _wb.FORK_WORKER_FLOOR,
                                        'parallel_workers_capped_by_budget')
    if n_fit < 2:
        return None
    if n_fit < n_w:
        n_w = n_fit
        size = max(1, -(-n // (4 * n_w)))
        chunks = [list(range(a, min(n, a + size))) for a in range(0, n, size)]
    _FANOUT['fns'], _FANOUT['tau_list'] = fns, tau_list
    try:
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_w, **_worker_share(n_w)) as pool:
            parts = _wb.watched_map(pool, _fork_task, chunks, share)
    finally:
        _FANOUT.clear()
    if parts is None:               # a worker outgrew its share: serial
        return None
    vals = [None] * n
    for out, delta in parts:
        for d, v in out:
            vals[d] = v
        _fi.merge_counter_delta(delta)
    _fi._RUNTIME_COUNTERS['parallel_fork_batches'] = (
        _fi._RUNTIME_COUNTERS.get('parallel_fork_batches', 0) + 1)
    return vals


def _ss_grid(points):
    """The state-space route's τ batch over ``points`` (the raw tuples
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
    memory budget: a worker builds its own copy of the lazily
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


def _worker_budget_init(share):
    """Pool initializer of a forked Phase J batch: the worker's running
    total may grow by its ``share`` of what the parent's budget had left
    (each worker builds its own lazily built states and charges them to its
    own copy of the total; with the full budget each, n workers could
    together use n times it)."""
    from engine.integration.time_domain import final_integral as _fi
    lock = getattr(_fi, '_PLAN_MEM_LOCK', None)
    if lock is None or share is None:
        return
    with lock:
        if _fi._PLAN_MEM['active'] > 0 and _fi._PLAN_MEM['budget'] is not None:
            _fi._PLAN_MEM['budget'] = _fi._PLAN_MEM['used'] + share


def _worker_share(n_w):
    """``(initializer, initargs)`` giving each of ``n_w`` forked workers an
    equal share of the running budget's remainder (no limit: no
    initializer)."""
    from engine.integration.time_domain import final_integral as _fi
    lock = getattr(_fi, '_PLAN_MEM_LOCK', None)
    if lock is None:
        return {}
    with lock:
        if _fi._PLAN_MEM['active'] <= 0 or _fi._PLAN_MEM['budget'] is None:
            return {}
        left = max(0, _fi._PLAN_MEM['budget'] - _fi._PLAN_MEM['used'])
    return {'initializer': _worker_budget_init,
            'initargs': (left // max(1, int(n_w)),)}


class DiscardedDiagramsError(DaedalusError, RuntimeError):
    """A callable of a Phase J run made with ``discard=True`` was
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
    (the diagram's closure has been dropped)."""
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


def _mem_delta(mark):
    """``(bytes, tuples, plans)`` charged to the running budget since
    ``mark`` (zeros outside a scope)."""
    from engine.integration.time_domain import final_integral as _fi
    scoped, used0, tuples0, plans0 = mark
    with _fi._PLAN_MEM_LOCK:
        m = _fi._PLAN_MEM
        if not scoped or m['active'] <= 0:
            return (0, 0, 0)
        return (m['used'] - used0, m['tuples'] - tuples0,
                m['plans'] - plans0)


def _mem_refund(mark, charge):
    """Take ``charge`` (``_mem_delta``) back from the running total: the
    charge of a diagram released after other diagrams were charged."""
    from engine.integration.time_domain import final_integral as _fi
    with _fi._PLAN_MEM_LOCK:
        m = _fi._PLAN_MEM
        if not mark[0] or m['active'] <= 0:
            return
        m['used'] -= charge[0]
        m['tuples'] -= charge[1]
        m['plans'] -= charge[2]


def _growth_over(fraction):
    """Has the running ``compute_cumulants`` grown by more than ``fraction``
    of its budget (``final_integral.run_growth``)?"""
    from engine.integration.time_domain import final_integral as _fi
    rg = getattr(_fi, 'run_growth', None)
    if rg is None:
        return False
    growth, budget = rg()
    return (growth is not None and budget is not None
            and growth > fraction * budget)


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
    # retired: raises; pass typed_diagrams + prefactors
    kernel_groups=None,
    # M8: divergence policy of the box-free routes (``integrate_diagram``)
    on_divergence=None,
    # evaluate each diagram on these points right after it is
    # built, keep the values, release it (``discard``)
    discard_points=None,
    # evaluate them under the state-space τ batch (k <= 2: as
    # ``total_C_batch`` does) or point by point (k >= 3: as a call does)
    discard_prefetch=True,
    # 'auto' mid-run: keep the callables until the run has grown by this
    # fraction of its budget, then evaluate and release
    discard_defer=None,
):
    r"""
    Time-domain entry point: evaluate typed diagrams by Phase J vertex-time
    integration (closed forms on numeric poles; see
    ``final_integral.integrate_diagram``).

    Takes ``typed_diagrams`` + ``prefactors`` (one prefactor per diagram).
    The legacy ``kernel_groups=`` input is retired: it integrated every
    diagram of a group with the group's summed prefactor and
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
        evaluate-and-discard: each diagram is evaluated on these
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
    discard_prefetch : bool, default True
        With ``discard_points``: evaluate each diagram on them under the
        state-space τ batch (``True``; the same values as ``total_C_batch``)
        or one call per point (``False``; the same values as calling
        ``total_C`` at each point, which is how k >= 3 results are read).
    discard_defer : float or None
        With ``discard_points``: start with the callables kept (as without
        ``discard_points``) and switch to evaluate-and-discard at the first
        diagram boundary where the run's measured growth
        (``final_integral.run_growth``) exceeds this fraction of its budget.
        The diagrams built before then are evaluated on the points in
        diagram order and released at the switch, so the sums are the same.
        Without a switch the result is the one without ``discard_points``;
        ``result['discard']['switched_at']`` is the first diagram built in
        discard mode.
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
    # evaluate-and-discard from the first diagram, or (``discard_defer``)
    # from the switch on
    discarding = discard_points is not None and discard_defer is None
    discard_vals = []           # per kept diagram: complex values per point
    discard_info = {'points': discard_points, 'diagrams': 0, 'refunded': 0,
                    'kept_for_delta': 0, 'built_bytes': 0,
                    'max_diagram_bytes': 0,
                    'switched_at': 0 if discarding else None}
    # ``discard_defer`` before the switch: (index into tree_callables, into
    # groups_out, diagram index, its charge, has δ records) per diagram
    backlog = []

    def _eval_points(fn):
        if discard_prefetch:
            with _ss_grid(discard_points):
                return [complex(fn(*pt)) for pt in discard_points]
        return [complex(fn(*pt)) for pt in discard_points]

    def _count(nb, has_delta):
        discard_info['diagrams'] += 1
        discard_info['built_bytes'] += nb
        discard_info['max_diagram_bytes'] = max(
            discard_info['max_diagram_bytes'], nb)
        if has_delta:
            discard_info['kept_for_delta'] += 1
        else:
            discard_info['refunded'] += 1

    from engine.integration.time_domain import final_integral as _fi_g
    _growth = getattr(_fi_g, 'growth_guard', None)
    for idx, (td, pf) in enumerate(zip(typed_diagrams, prefactors)):
        loop_number = _loop_number_from_graph(td)
        if _growth is not None:
            _growth(f'diagram #{idx} of {len(typed_diagrams)}')
        if (discard_points is not None and not discarding
                and _growth_over(discard_defer)):
            # the switch: the diagrams kept so far are evaluated on the
            # points (in diagram order) and released
            for ti, gi, didx, charge, has_delta in backlog:
                mark_b = _mem_mark()
                discard_vals.append(_eval_points(tree_callables[ti]))
                charge = tuple(a + b for a, b in
                               zip(charge, _mem_delta(mark_b)))
                released = _discarded_callable(didx)
                tree_callables[ti] = released
                groups_out[gi]['contribution'] = released
                if not has_delta:
                    _mem_refund(mark_b, charge)
                _count(charge[0], has_delta)
            backlog = []
            discarding = True
            discard_info['switched_at'] = idx
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
            has_delta = bool(result.get('delta_contributions')
                             or result.get('delta_contributions_multi'))
            if discarding:
                discard_vals.append(_eval_points(contribution))
                contribution = _discarded_callable(idx)
                result['contribution'] = contribution
                _count(_mem_release(mark, refund=not has_delta), has_delta)
            elif discard_points is not None:
                backlog.append((len(tree_callables), len(groups_out), idx,
                                _mem_delta(mark), has_delta))
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

    if discarding:
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
                          start_method=None):
            return [total_C(*pt) for pt in tau_points]

        def eval_per_diagram_batch(tau_points, parallel=True, n_workers=None,
                                   start_method=None):
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
                      start_method=None):
        """``total_C`` on a list of τ points, in input order.

        ``parallel=True`` fans the diagrams out to a worker pool when one
        helps (module notes: threads for diagrams with large state blocks,
        on every platform; fork on Linux; never fork on macOS) and sums each
        point's diagram values in diagram order, so the result is
        bit-identical to ``parallel=False`` for any ``n_workers``.
        ``n_workers``: pool size (default: up to ``DEFAULT_THREADS``
        threads, ``os.cpu_count()`` fork workers, within the memory budget).
        ``start_method``: ``None`` (auto), ``'threads'`` or ``'fork'``
        (Linux; on macOS it falls back with one warning).
        ``DAEDALUS_PARALLEL=0`` makes every batch serial.
        """
        tau_list = [tuple(pt) for pt in tau_points]
        vals = _fan_out(tree_callables, tau_list, parallel, n_workers,
                        start_method)
        if vals is None:
            with _ss_grid(tau_list):
                return [total_C(*pt) for pt in tau_list]
        out = []
        for p_idx in range(len(tau_list)):
            s = 0.0 + 0.0j
            for v in vals:
                s = s + v[p_idx]
            out.append(s)
        return out

    def eval_per_diagram_batch(tau_points, parallel=True, n_workers=None,
                                start_method=None):
        """Evaluate EACH diagram's contribution at EACH τ point, returning
        the full (n_tau, n_diagram) grid (same parallelism as
        ``total_C_batch``).

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
        loop_nums = [g.get('loop_number', 0) for g in groups_out
                     if g.get('contribution') is not None]
        if not tree_callables or not tau_list:
            return [[] for _ in tau_list], loop_nums
        vals = _fan_out(tree_callables, tau_list, parallel, n_workers,
                        start_method)
        if vals is None:
            rows = []
            with _ss_grid(tau_list):
                for pt in tau_list:
                    rows.append([complex(fn(*pt)) for fn in tree_callables])
            return rows, loop_nums
        return ([[v[p_idx] for v in vals] for p_idx in range(len(tau_list))],
                loop_nums)

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

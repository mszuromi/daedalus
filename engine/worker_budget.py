"""
engine.worker_budget
===================
Memory caps for every worker pool, on top of the run's memory
budget and growth baseline (``final_integral.run_growth_scope`` /
``run_budget``):

* :func:`budget_left` -- what the run's budget still allows: the budget
  minus the process's growth since the run started (``None``: no limit).
* :func:`workers_that_fit` -- how many workers of an estimated size fit in
  it (fewer workers, then 1 = serial; never more than asked).
* :class:`FootprintWatch` -- a thread that polls the workers' own memory
  (macOS ``phys_footprint`` through ``proc_pid_rusage``; Linux the private
  pages of ``/proc/<pid>/smaps_rollup``, so the pages a forked child still
  shares with its parent are not counted twice) and reports a worker over
  its share (``on_over``: the pool kills it or stops, and the work it held
  is redone in the parent, under the parent's own growth guard).  Never a
  swap storm: a pool can only shrink.
"""
import os
import sys
import threading

#: Footprint of a fresh Sage-free spawned worker before any diagram
#: (measured 37 MB for numpy + scipy; 64 MB leaves room for the modules).
SPAWN_WORKER_BASE = 64 * 2**20
#: Smallest share a forked Sage worker gets (its private pages grow as it
#: types diagrams or builds states).
FORK_WORKER_FLOOR = 256 * 2**20
#: Seconds between two polls of the workers' footprints.
POLL_SECONDS = 0.25


def pid_footprint_bytes(pid):
    """Memory held by process ``pid`` alone, in bytes, or ``None``."""
    if sys.platform == 'darwin':
        try:
            import ctypes
            from engine.integration.time_domain.final_integral import \
                _libproc
            buf = ctypes.create_string_buffer(512)
            if _libproc().proc_pid_rusage(int(pid), 2, buf) != 0:
                return None
            return int.from_bytes(buf.raw[16 + 7 * 8:16 + 8 * 8], 'little')
        except Exception:
            return None
    try:
        total = 0
        with open(f'/proc/{int(pid)}/smaps_rollup') as fh:
            for line in fh:
                if line.startswith(('Private_Clean:', 'Private_Dirty:',
                                    'Swap:')):
                    total += int(line.split()[1]) * 1024
        return total or None
    except (OSError, ValueError, IndexError):
        return None


def budget_left():
    """``(bytes left or None, rule)``: the running budget minus the growth
    of this process since the run's ``run_growth_scope`` began (outside a
    scope: the budget minus nothing).  ``None``: no limit."""
    from engine.integration.time_domain import final_integral as FI
    budget, rule = FI.run_budget()
    if budget is None:
        return None, rule
    grown = 0
    with FI._PLAN_MEM_LOCK:
        fp0 = FI._RUN_GROWTH['fp0'] if FI._RUN_GROWTH['active'] > 0 else None
    if fp0 is not None:
        fp = FI.process_footprint_bytes()
        if fp is not None:
            grown = max(0, fp - fp0)
    return max(0, budget - grown), rule


def workers_that_fit(n, per_worker, counter=None):
    """``(n_fit, share)``: at most ``n`` workers of ``per_worker`` bytes each
    within :func:`budget_left` (at least 1: the caller runs serially), and
    each worker's share of it (``None``: no limit).  A cap is counted in
    ``_RUNTIME_COUNTERS[counter]``."""
    left, _rule = budget_left()
    n = max(1, int(n))
    if left is None:
        return n, None
    fit = int(left // max(1, int(per_worker)))
    n_fit = max(1, min(n, fit))
    if n_fit < n and counter:
        from engine.integration.time_domain import final_integral as FI
        FI._RUNTIME_COUNTERS[counter] = FI._RUNTIME_COUNTERS.get(counter, 0) + 1
    return n_fit, left // n_fit


class FootprintWatch:
    """Polls ``pids()`` every ``POLL_SECONDS``; a pid over ``share`` bytes is
    passed to ``on_over(pid, footprint)`` once.  ``relative=True`` measures
    each pid's growth since its first reading (a forked child: what it holds
    from the start is pages it may still share with its parent, dirtied by
    reference counting).  ``share=None``: no limit (the watch does
    nothing).  Use as a context manager."""

    def __init__(self, pids, share, on_over,
                 counter='parallel_workers_over_share', relative=False):
        self.pids, self.share, self.on_over = pids, share, on_over
        self.counter, self.relative = counter, relative
        self.over = {}
        self.peak = {}
        self.first = {}
        self._stop = threading.Event()
        self._th = None

    def _run(self):
        while not self._stop.wait(POLL_SECONDS):
            self.poll()

    def poll(self):
        for pid in list(self.pids()):
            fp = pid_footprint_bytes(pid)
            if fp is None:
                continue
            self.peak[pid] = max(self.peak.get(pid, 0), fp)
            if self.relative:
                fp = fp - self.first.setdefault(pid, fp)
            if self.share is not None and fp > self.share \
                    and pid not in self.over:
                self.over[pid] = fp
                from engine.integration.time_domain import final_integral as FI
                FI._RUNTIME_COUNTERS[self.counter] = (
                    FI._RUNTIME_COUNTERS.get(self.counter, 0) + 1)
                try:
                    self.on_over(pid, fp)
                except Exception:                   # pragma: no cover
                    pass

    def __enter__(self):
        if self.share is not None:
            self._th = threading.Thread(target=self._run, daemon=True,
                                        name='daedalus-footprint-watch')
            self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5)
        return False


def kill_pid(pid):
    try:
        import signal
        os.kill(int(pid), signal.SIGKILL)
    except (OSError, ValueError):
        pass


def watched_map(pool, fn, items, share):
    """``pool.map(fn, items)`` (a ``multiprocessing`` pool) while a
    :class:`FootprintWatch` polls its workers; ``None`` -- the pool
    terminated -- as soon as one outgrows ``share``: the caller then does the
    work serially, under its own growth guard."""
    over = threading.Event()
    res = pool.map_async(fn, items, chunksize=1)
    start = [p for p in list(getattr(pool, '_pool', ()))]

    def pids():
        return [p.pid for p in list(getattr(pool, '_pool', ())) if p.pid]
    with FootprintWatch(pids, share, lambda pid, fp: over.set(),
                        relative=True):
        while not res.ready():
            res.wait(0.2)
            # a worker that died (killed from outside, e.g. by the OS) never
            # returns its task: multiprocessing would wait for ever
            dead = any(p.exitcode is not None for p in start)
            if over.is_set() or dead:
                pool.terminate()
                return None
    return res.get()

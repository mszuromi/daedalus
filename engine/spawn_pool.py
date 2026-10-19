"""
engine.spawn_pool
================
Parent side of the spawned Phase J workers (``engine._spawn_worker``): a
small pool of fresh interpreters for the platforms where fork is not
allowed (macOS never forks; Windows cannot).

* Each worker is ``python -m engine._spawn_worker`` started with
  ``subprocess`` (exec at once, no ``__main__`` re-import) in its own session
  (a Ctrl-C in the parent's terminal or kernel does not reach it; the parent
  stops it), with ``OPENBLAS_NUM_THREADS`` / ``OMP_NUM_THREADS`` /
  ``MKL_NUM_THREADS`` / ``VECLIB_MAXIMUM_THREADS`` = 1 in ITS environment
  only, and the package root on its ``PYTHONPATH``.  It never imports Sage.
* All workers start at once; then each gets the objects the tasks share and
  the parent's settings (``init``).  One driver thread per worker sends tasks
  (largest first) and collects the values.
* A task that a worker could not evaluate, a worker that dies or answers
  anything unexpected, never loses a diagram: :meth:`SpawnPool.map` reports
  those indices and the caller evaluates them in this process.
* :meth:`SpawnPool.close` (also on any error or interrupt) asks every worker
  to exit and kills it after a short grace period.
"""
import io
import os
import pickle
import subprocess
import sys
import threading
import time

from engine._spawn_worker import SharedPickler, read_frame, write_frame

#: Environment of every worker: one BLAS / OpenMP thread each.
WORKER_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpawnWorkerError(RuntimeError):
    """A worker could not be started or initialised."""


def _popen(env):
    kw = {}
    if os.name == 'posix':
        kw['start_new_session'] = True
    else:                                           # pragma: no cover
        kw['creationflags'] = getattr(subprocess,
                                      'CREATE_NEW_PROCESS_GROUP', 0)
    return subprocess.Popen(
        [sys.executable, '-m', 'engine._spawn_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        env=env, cwd=_ROOT, close_fds=True, **kw)


class _Worker:
    def __init__(self, proc):
        self.proc = proc
        self.alive = True
        self.info = {}

    def handshake(self, shared_bytes, settings):
        try:
            write_frame(self.proc.stdin, ('init', shared_bytes, settings))
            msg = read_frame(self.proc.stdout)
        except Exception as exc:
            self.kill()
            raise SpawnWorkerError(
                f'a Phase J worker process did not start '
                f'(exit code {self.proc.poll()}): {exc}') from None
        if not (isinstance(msg, tuple) and msg and msg[0] == 'ready'):
            self.kill()
            raise SpawnWorkerError(f'unexpected worker reply {msg!r:.80}')
        self.info = msg[1]

    def kill(self):
        self.alive = False
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:                           # pragma: no cover
            pass
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except Exception:
                pass


class SpawnPool:
    """``n`` spawned workers sharing ``shared`` (``{id: obj}``), set up with
    ``settings`` (``engine._spawn_worker``: the parent's budget, module
    settings)."""

    def __init__(self, n, shared, settings=None):
        env = dict(os.environ)
        env.update(WORKER_ENV)
        env['PYTHONPATH'] = os.pathsep.join(
            [_ROOT] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        settings = dict(settings or {})
        settings.setdefault('budget', (None, 'none'))
        self._shared_ids = set(shared)
        self.workers = []
        self.sage_loaded = False
        self.t_start = time.perf_counter()
        try:
            shared_bytes = pickle.dumps(shared,
                                        protocol=pickle.HIGHEST_PROTOCOL)
            for _ in range(int(n)):                 # start them all ...
                self.workers.append(_Worker(_popen(env)))
            for w in self.workers:                  # ... then initialise
                w.handshake(shared_bytes, settings)
        except BaseException:
            self.close()
            raise
        self.t_ready = time.perf_counter() - self.t_start
        self.sage_loaded = any(w.info.get('sage_loaded')
                               for w in self.workers)

    def pids(self):
        return [w.proc.pid for w in self.workers if w.alive]

    def encode(self, tasks):
        """The task pickles (shared objects by id).  Done in the calling
        thread, before anything else touches the payloads."""
        blobs = []
        for chunk in tasks:
            buf = io.BytesIO()
            SharedPickler(buf, self._shared_ids).dump(chunk)
            blobs.append(buf.getvalue())
        return blobs

    def map(self, tasks, points, blobs=None):
        """Evaluate ``tasks`` (a list of ``[(index, payload), ...]`` chunks,
        largest first) at ``points``.  Returns ``(values, states, counters,
        lost)``: ``values[index]`` for every evaluated diagram, the
        stationary states built in the workers (``states[index]``: a list of
        ``(subset, states, residual)``), the workers' counter deltas in task
        order, and the indices left to the caller (refused in a worker, or
        on a worker that failed)."""
        if blobs is None:
            blobs = self.encode(tasks)
        lock = threading.Lock()
        nxt = [0]
        values, states, lost = {}, {}, []
        counters_by_task = [None] * len(tasks)

        def drive(w):
            while True:
                with lock:
                    t = nxt[0]
                    if t >= len(tasks):
                        return
                    nxt[0] += 1
                try:
                    write_frame(w.proc.stdin, ('task', t, blobs[t], points))
                    msg = read_frame(w.proc.stdout)
                    ok = (isinstance(msg, tuple) and len(msg) == 5
                          and msg[0] == 'done' and msg[1] == t)
                except Exception:
                    ok = False
                if not ok:
                    # dead, killed (over its share) or garbled: stop it, give
                    # this task back; the queue goes on with the others
                    w.kill()
                    with lock:
                        lost.extend(i for i, _ in tasks[t])
                    return
                with lock:
                    for i, v, st in msg[2]:
                        if v is None:
                            lost.append(i)
                        else:
                            values[i] = v
                            if st:
                                states[i] = st
                    counters_by_task[t] = msg[3]
                    if msg[4]:
                        self.sage_loaded = True

        threads = [threading.Thread(target=drive, args=(w,), daemon=True,
                                    name=f'daedalus-spawn-{w.proc.pid}')
                   for w in self.workers if w.alive]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # tasks no live worker reached (every worker died)
        with lock:
            for t in range(nxt[0], len(tasks)):
                lost.extend(i for i, _ in tasks[t])
        counters = [c for c in counters_by_task if c]
        return values, states, counters, sorted(set(lost))

    def close(self):
        for w in self.workers:
            if w.alive:
                try:
                    write_frame(w.proc.stdin, ('exit',))
                except Exception:
                    pass
        deadline = time.perf_counter() + 2.0
        for w in self.workers:
            try:
                w.proc.wait(timeout=max(0.0, deadline - time.perf_counter()))
            except Exception:
                pass
            w.kill()
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

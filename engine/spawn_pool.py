"""
engine.spawn_pool
================
Parent side of the spawned Phase J workers (``engine._spawn_worker``): a
small pool of fresh interpreters for the platforms where fork is not
allowed (macOS never forks; Windows cannot).

* Each worker is ``python -m engine._spawn_worker`` started with
  ``subprocess`` (exec at once, no ``__main__`` re-import), with
  ``OPENBLAS_NUM_THREADS`` / ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
  ``VECLIB_MAXIMUM_THREADS`` = 1 in ITS environment only, and the package
  root on its ``PYTHONPATH``.  It never imports Sage.
* The objects the tasks share travel once per worker; one driver thread per
  worker sends tasks (largest first) and collects the values.
* A task that a worker could not evaluate, or a worker that dies, never
  loses a diagram: :meth:`SpawnPool.map` reports those indices and the
  caller evaluates them in this process.
* :meth:`SpawnPool.close` (also on any error or interrupt) asks every worker
  to exit and kills it after a short grace period.
"""
import io
import os
import subprocess
import sys
import threading
import time

from engine._spawn_worker import (SharedPickler, read_frame, write_frame)

#: Environment of every worker: one BLAS / OpenMP thread each.
WORKER_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpawnWorkerError(RuntimeError):
    """A worker could not be started or initialised."""


class _Worker:
    def __init__(self, env, shared_bytes):
        self.proc = subprocess.Popen(
            [sys.executable, '-m', 'engine._spawn_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=env, cwd=_ROOT, close_fds=True)
        self.alive = True
        try:
            write_frame(self.proc.stdin, ('init', shared_bytes))
            msg = read_frame(self.proc.stdout)
        except (EOFError, OSError, BrokenPipeError) as exc:
            self.kill()
            raise SpawnWorkerError(
                f'a Phase J worker process did not start '
                f'(exit code {self.proc.poll()}): {exc}') from None
        if msg[0] != 'ready':
            self.kill()
            raise SpawnWorkerError(f'unexpected worker reply {msg[0]!r}')
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
    """``n`` spawned workers sharing ``shared`` (``{id: obj}``)."""

    def __init__(self, n, shared):
        env = dict(os.environ)
        env.update(WORKER_ENV)
        env['PYTHONPATH'] = os.pathsep.join(
            [_ROOT] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        import pickle
        self._shared_ids = set(shared)
        shared_bytes = pickle.dumps(shared, protocol=pickle.HIGHEST_PROTOCOL)
        self.workers = []
        self.t_start = time.perf_counter()
        try:
            # start them all, then wait for each one's 'ready'
            for _ in range(int(n)):
                self.workers.append(_Worker(env, shared_bytes))
        except BaseException:
            self.close()
            raise
        self.t_ready = time.perf_counter() - self.t_start
        self.sage_loaded = any(w.info.get('sage_loaded') for w in self.workers)

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

    def map(self, tasks, points, blobs=None, on_result=None):
        """Evaluate ``tasks`` (a list of ``[(index, payload), ...]`` chunks,
        largest first) at ``points``.  Returns ``(values, counters, lost)``:
        ``values[index]`` for every evaluated diagram, the workers' counter
        deltas in task order, and the indices left to the caller (refused in
        a worker, or on a worker that failed)."""
        if blobs is None:
            blobs = self.encode(tasks)
        lock = threading.Lock()
        nxt = [0]
        values, lost = {}, []
        counters_by_task = [None] * len(tasks)
        errors = []

        def drive(w):
            while True:
                with lock:
                    if errors:
                        return
                    t = nxt[0]
                    if t >= len(tasks):
                        return
                    nxt[0] += 1
                try:
                    write_frame(w.proc.stdin, ('task', t, blobs[t], points))
                    msg = read_frame(w.proc.stdout)
                except (EOFError, OSError, BrokenPipeError):
                    w.alive = False
                    with lock:
                        lost.extend(i for i, _ in tasks[t])
                        # the rest of the queue goes to the live workers
                    return
                with lock:
                    if msg[0] == 'done':
                        for i, v in msg[2]:
                            if v is None:
                                lost.append(i)
                            else:
                                values[i] = v
                        counters_by_task[t] = msg[3]
                    else:
                        lost.extend(i for i, _ in tasks[t])
                if on_result is not None:
                    on_result()

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
        return values, counters, sorted(set(lost))

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

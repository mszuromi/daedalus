"""
engine._spawn_worker
===================
A Phase J worker PROCESS for parallel τ batches where fork is not allowed
(macOS, Windows).  The parent (``engine.spawn_pool``) starts it as a fresh
interpreter -- ``python -m engine._spawn_worker`` through ``subprocess``,
which execs at once: nothing of the parent's state (Cocoa, BLAS pools, ECL
threads, a Jupyter kernel) is inherited, and the user's ``__main__`` is never
re-imported (unlike ``multiprocessing``'s spawn).

The worker never imports Sage.  It loads numpy, scipy and the Sage-free
modules it evaluates with (``time_domain._light``, ``time_domain.statespace``,
``api._realization``), with bare stand-ins for their parent packages
(``engine.integration.time_domain`` and ``api``, whose ``__init__`` files
import Sage); see :func:`_bootstrap`.  A diagram task is the numeric state
the parent already built (``statespace.diagram_payload``), so the worker's
values are the same bytes as the parent's.

Protocol (binary stdin/stdout, frames of an 8-byte little-endian length and a
pickle):
  parent -> worker  ('init', shared_bytes)       objects shared by the tasks
                    ('task', tid, task_bytes, points)
                    ('exit',)
  worker -> parent  ('ready', info)              after init: pid, Sage loaded?
                    ('done', tid, results, counters)
                       results: [(index, values or None)]; None = the
                       diagram raised here (the parent evaluates it itself)
                    ('fail', tid, message)       the task itself failed
Shared objects (model / entry ops) travel once; tasks reference them by the
parent's ``id()`` (pickle persistent ids).
"""
import os
import pickle
import struct
import sys
import types

_LEN = struct.Struct('<Q')


def write_frame(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_LEN.pack(len(data)))
    stream.write(data)
    stream.flush()


def read_frame(stream):
    head = stream.read(_LEN.size)
    if len(head) < _LEN.size:
        raise EOFError('worker stream closed')
    (n,) = _LEN.unpack(head)
    data = stream.read(n)
    if len(data) < n:
        raise EOFError('worker stream closed mid-frame')
    return pickle.loads(data)


class SharedPickler(pickle.Pickler):
    """Pickles a task, replacing the shared objects by their ids."""

    def __init__(self, file, shared_ids):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._ids = shared_ids

    def persistent_id(self, obj):
        i = id(obj)
        return ('shared', i) if i in self._ids else None


class SharedUnpickler(pickle.Unpickler):
    def __init__(self, file, shared):
        super().__init__(file)
        self._shared = shared

    def persistent_load(self, pid):
        kind, i = pid
        if kind != 'shared':
            raise pickle.UnpicklingError(f'unknown persistent id {pid!r}')
        return self._shared[i]


def _stub_package(name, path):
    """A bare package module ``name`` whose submodules load from ``path``
    (its own ``__init__`` is not run)."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition('.')
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)


def _bootstrap():
    """Make the evaluation modules importable without Sage (worker only)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import engine                                   # empty __init__
    import engine.integration                       # empty __init__
    _stub_package('engine.integration.time_domain',
                  os.path.join(root, 'engine', 'integration', 'time_domain'))
    _stub_package('api', os.path.join(root, 'api'))
    from engine.integration.time_domain import statespace, _light
    return statespace, _light


def _serve(stdin, stdout):
    import io
    statespace, _light = _bootstrap()
    msg = read_frame(stdin)
    if msg[0] != 'init':
        raise RuntimeError(f'expected init, got {msg[0]!r}')
    shared = pickle.loads(msg[1])
    write_frame(stdout, ('ready', {
        'pid': os.getpid(),
        'sage_loaded': any(m == 'sage' or m.startswith('sage.')
                           for m in sys.modules)}))
    while True:
        msg = read_frame(stdin)
        if msg[0] == 'exit':
            return
        if msg[0] != 'task':
            raise RuntimeError(f'unknown message {msg[0]!r}')
        _, tid, blob, points = msg
        try:
            items = SharedUnpickler(io.BytesIO(blob), shared).load()
        except Exception as exc:                    # pragma: no cover
            write_frame(stdout, ('fail', tid, f'{type(exc).__name__}: {exc}'))
            continue
        results, counters = [], {}
        for idx, payload in items:
            try:
                with _light.counter_delta() as cd:
                    vals = statespace.payload_values(payload, points)
            except Exception:
                results.append((idx, None))   # the parent evaluates it
                continue
            results.append((idx, vals))
            for k, v in cd.delta.items():     # this diagram's counters
                if isinstance(v, dict):
                    d = counters.setdefault(k, {})
                    for r, n in v.items():
                        d[r] = d.get(r, 0) + n
                elif k.endswith('_max'):
                    counters[k] = max(counters.get(k, 0), v)
                else:
                    counters[k] = counters.get(k, 0) + v
        write_frame(stdout, ('done', tid, results, counters))


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # anything printed by imported code must not corrupt the frame stream
    sys.stdout = sys.stderr
    try:
        _serve(stdin, stdout)
    except EOFError:
        pass                        # the parent went away: just stop


if __name__ == '__main__':
    main()

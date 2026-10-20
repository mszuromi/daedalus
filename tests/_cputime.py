"""CPU-time clock for the speed-bound tests.

A wall-clock limit fails on a loaded machine (several test processes, a paper-example
pipeline) although the code did the same work.  ``CpuClock`` counts the CPU seconds this
test process used (all its threads), plus those of its child processes (Sage's Maxima
helper does the symbolic work of several of these tests): children that exited and were
reaped through ``os.times()``, the live ones through ``psutil`` when it is installed.
Without psutil the live children (a persistent Maxima) are not seen, so a limit then
only bounds the Python-side work -- never stricter than the wall clock.
"""
from __future__ import annotations

import os
import time

try:                                    # optional
    import psutil
except ImportError:                     # pragma: no cover
    psutil = None


def _children_live():
    if psutil is None:
        return {}
    out = {}
    try:
        for c in psutil.Process().children(recursive=True):
            try:
                t = c.cpu_times()
                out[c.pid] = t.user + t.system
            except psutil.Error:
                pass
    except psutil.Error:                # pragma: no cover
        pass
    return out


def _snapshot():
    t = os.times()
    return time.process_time(), t.children_user + t.children_system, _children_live()


class CpuClock:
    """``clock = CpuClock(); ...; clock.elapsed()`` -> CPU seconds since construction
    (or the last ``reset()``)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._p0, self._r0, self._live0 = _snapshot()

    def elapsed(self):
        p, r, live = _snapshot()
        live_delta = sum(max(0.0, v - self._live0.get(pid, 0.0)) for pid, v in live.items())
        return (p - self._p0) + (r - self._r0) + live_delta

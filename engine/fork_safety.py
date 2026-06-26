"""
engine.fork_safety
=================
Canonical fork-safety guard, shared by every fork-based multiprocessing path.

Forking a process AFTER Cocoa / BLAS / matplotlib initialization inside a
Jupyter kernel on macOS can hard-crash the kernel AND the OS (it has crashed
this project's dev machine more than once).  Setting
``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` makes it WORSE, not better — it
turns a controlled abort into a hard crash.

So every code path that is about to ``multiprocessing.get_context('fork')``
must first consult :func:`fork_unsafe_in_notebook` and degrade to serial when it
returns True.  Fork stays available — and fast — for plain scripts
(``sage -python run.py``), pytest, Linux, and terminal IPython, none of which
are exposed to the fork-after-Cocoa-init crash.  Thread-based parallelism (the
spatial path) needs no guard.
"""
import sys
import warnings

__all__ = ['fork_unsafe_in_notebook', 'warn_fork_fallback_once']

_WARNED = set()


def fork_unsafe_in_notebook(start_method='fork'):
    """Return True iff a ``fork`` here risks crashing a macOS Jupyter kernel.

    Narrow by design: ``start_method == 'fork'`` AND ``sys.platform ==
    'darwin'`` AND we are inside a ZMQ/Jupyter interactive kernel.  Returns
    False for spawn/forkserver, non-macOS, terminal IPython, plain scripts, and
    pytest — none of which are exposed to the fork-after-Cocoa-init crash.
    """
    if start_method != 'fork':
        return False
    if sys.platform != 'darwin':
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return False
    return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'


def warn_fork_fallback_once(where):
    """Emit the fork-in-notebook → serial fallback warning at most once per
    ``where`` (a short label for the call site) per process."""
    if where in _WARNED:
        return
    _WARNED.add(where)
    warnings.warn(
        f"{where}: fork-based multiprocessing is UNSAFE inside a Jupyter kernel "
        "on macOS — forking after Cocoa/BLAS init can hard-crash the kernel and "
        "the OS.  Automatically falling back to SERIAL evaluation.  To "
        "parallelise, run as a plain script (`sage -python run.py`, not a "
        "notebook), where fork-after-init is safe.",
        RuntimeWarning,
        stacklevel=3,
    )

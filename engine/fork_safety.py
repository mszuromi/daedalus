"""
engine.fork_safety
=================
Canonical fork-safety guard and parallel kill switch, shared by every worker
pool of the package (Phase J tau batches, type assignment, streaming
enumeration, the legacy eager enumerator, spatial threads).

**macOS never forks.**  Forking a process after Cocoa / BLAS / matplotlib
initialization inside a Jupyter kernel on macOS has hard-crashed the kernel AND
the OS of this project's dev machine more than once.  Plain scripts survived
only because AppKit never loads there; a Sage process is multithreaded after
its first symbolic step (Boehm GC marker threads of ECL), so the fork parent is
never clean.  :func:`fork_unsafe` is therefore True for ``fork`` on every macOS
process, notebook or not, and each fork-based pool degrades to serial with one
warning per process (:func:`warn_fork_fallback_once`).  Fork stays available on
Linux.  ``spawn`` / ``forkserver`` are never reported unsafe here.

``OBJC_DISABLE_INITIALIZE_FORK_SAFETY`` is never set by the package: the
Objective-C runtime reads it once at image load, so setting it from Python had
no effect on the forks it was meant to guard, and it leaked into every process
exec'd later, where it turns the runtime's controlled abort into a hard crash.

**Kill switch.**  ``DAEDALUS_PARALLEL=0`` (also false/no/off) makes every pool
serial, threads included, silently; read at call time (:func:`parallel_disabled`).
"""
import os
import sys
import warnings

__all__ = ['fork_unsafe', 'fork_unsafe_in_notebook', 'in_jupyter_kernel',
           'parallel_disabled', 'warn_fork_fallback_once', 'PARALLEL_ENV']

PARALLEL_ENV = 'DAEDALUS_PARALLEL'
_OFF = ('0', 'false', 'no', 'off')
_ON = ('', '1', 'true', 'yes', 'on')

_WARNED = set()


def parallel_disabled(environ=None):
    """True iff ``DAEDALUS_PARALLEL`` turns every worker pool off.

    Unset or ``1``/true/yes/on: pools allowed; ``0``/false/no/off: serial
    everywhere.  Any other value raises ``ValueError`` (a typo must not
    silently leave parallelism on)."""
    env = os.environ if environ is None else environ
    raw = env.get(PARALLEL_ENV)
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in _OFF:
        return True
    if val in _ON:
        return False
    raise ValueError(
        f'{PARALLEL_ENV}={raw!r}: expected 0/1 (also false/true, no/yes, '
        f'off/on).')


def fork_unsafe(start_method='fork'):
    """Return True iff a ``fork`` start here is not allowed: ``start_method ==
    'fork'`` on macOS, in any process (notebook, terminal IPython, script,
    pytest).  False for spawn/forkserver and for fork on other platforms."""
    return start_method == 'fork' and sys.platform == 'darwin'


def fork_unsafe_in_notebook(start_method='fork'):
    """Return True iff a ``fork`` here would happen inside a macOS Jupyter
    kernel (the context that crashed the OS).  Kept for callers that need to
    tell the kernel case apart; the pools consult :func:`fork_unsafe`, which
    is True for fork on macOS everywhere.

    ``start_method == 'fork'`` AND ``sys.platform == 'darwin'`` AND
    :func:`in_jupyter_kernel`.
    """
    if start_method != 'fork':
        return False
    if sys.platform != 'darwin':
        return False
    return in_jupyter_kernel()


def in_jupyter_kernel():
    """True inside ANY Jupyter/ZMQ kernel, detected conservatively.

    Three independent signals, any one suffices: ``ipykernel`` is imported (every
    Jupyter kernel, including the SageMath and VS Code kernels); the shell's MRO
    contains ``ZMQInteractiveShell`` (the SageMath 10.x kernel uses the SUBCLASS
    ``SageZMQInteractiveShell``, Spyder ``SpyderShell``, which an exact
    class-name test missed); or the shell carries a ``kernel``.  A false
    positive only costs speed (serial), a false negative can crash the OS, so
    the test errs towards True.
    """
    if 'ipykernel' in sys.modules:
        return True
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return False
    if ip is None:
        return False
    if any(c.__name__ == 'ZMQInteractiveShell' for c in type(ip).__mro__):
        return True
    return getattr(ip, 'kernel', None) is not None


_in_jupyter_kernel = in_jupyter_kernel        # PR #54 name


def warn_fork_fallback_once(where):
    """Emit the fork -> serial fallback warning once per process (the first
    call site to fall back names itself; later ones stay quiet)."""
    if _WARNED:
        return
    _WARNED.add(where)
    ctx = ('inside a Jupyter kernel' if in_jupyter_kernel()
           else 'in this process')
    warnings.warn(
        f"{where}: fork-based multiprocessing is disabled on macOS ({ctx}) -- "
        "forking after Cocoa/BLAS/ECL initialisation can hard-crash the "
        "kernel and the OS.  Running SERIALLY; the results are the same.  "
        "Pass parallel=False to silence this warning, or set "
        f"{PARALLEL_ENV}=0 to turn every worker pool off.",
        RuntimeWarning,
        stacklevel=3,
    )


def _reset_warnings():
    """Tests: let the next fallback warn again."""
    _WARNED.clear()

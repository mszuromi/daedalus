"""Unit tests for the shared fork-safety guard and kill switch
(engine/fork_safety.py).

macOS never forks (``fork_unsafe``): forking after Cocoa/BLAS/ECL init has
hard-crashed a Jupyter kernel AND the OS, and a Sage process is multithreaded
after its first symbolic step, so every fork-based pool runs serially on
macOS.  ``fork_unsafe_in_notebook`` / ``in_jupyter_kernel`` still tell the
kernel case apart (fake shells below: plain Jupyter, SageMath, Spyder, VS
Code).  ``DAEDALUS_PARALLEL=0`` turns every pool off.

Run:  sage -python -m pytest tests/test_fork_safety.py -q
"""
import os
import re
import sys
import types
import warnings
from pathlib import Path

import pytest

import engine.fork_safety as fs
from engine.fork_safety import (fork_unsafe, fork_unsafe_in_notebook,
                                in_jupyter_kernel, parallel_disabled)

ROOT = Path(__file__).resolve().parents[1]


def _fake_ipython(shell_class_name, bases=(), **attrs):
    """Install a fake ``IPython`` module whose get_ipython() returns a shell
    whose class is named ``shell_class_name``."""
    shell = type(shell_class_name, bases, dict(attrs))()
    mod = types.SimpleNamespace(get_ipython=lambda: shell)
    return mod


@pytest.fixture(autouse=True)
def _no_ipykernel(monkeypatch):
    """Each case decides itself whether a kernel module is loaded."""
    monkeypatch.delitem(sys.modules, 'ipykernel', raising=False)


def test_non_fork_start_methods_never_unsafe():
    assert fork_unsafe_in_notebook('spawn') is False
    assert fork_unsafe_in_notebook('forkserver') is False


def test_non_darwin_fork_is_safe(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('ZMQInteractiveShell'))
    assert fork_unsafe_in_notebook('fork') is False


def test_darwin_terminal_ipython_is_not_the_notebook_case(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('TerminalInteractiveShell'))
    assert fork_unsafe_in_notebook('fork') is False
    assert fork_unsafe('fork') is True          # ...but macOS never forks


def test_darwin_no_ipython_is_not_the_notebook_case(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython', None)   # import → fails → False
    assert fork_unsafe_in_notebook('fork') is False
    assert fork_unsafe('fork') is True          # ...but macOS never forks


def test_darwin_jupyter_kernel_is_unsafe(monkeypatch):
    """The one lethal case: fork + macOS + a ZMQ/Jupyter kernel."""
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('ZMQInteractiveShell'))
    assert fork_unsafe_in_notebook('fork') is True
    # ...but spawn there is still fine:
    assert fork_unsafe_in_notebook('spawn') is False


def test_darwin_sage_kernel_subclass_is_unsafe(monkeypatch):
    """The SageMath 10.x kernel's shell is SageZMQInteractiveShell, a SUBCLASS
    of ZMQInteractiveShell; the old exact class-name test returned False here
    and let compute_cumulants fork inside the user's notebooks."""
    monkeypatch.setattr(sys, 'platform', 'darwin')
    base = type('ZMQInteractiveShell', (), {})
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('SageZMQInteractiveShell', (base,)))
    assert fork_unsafe_in_notebook('fork') is True


def test_darwin_ipykernel_loaded_is_unsafe(monkeypatch):
    """Any loaded ipykernel (VS Code, Sage, plain Jupyter) counts, even when the
    shell itself is not recognisable."""
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython', _fake_ipython('SomeShell'))
    monkeypatch.setitem(sys.modules, 'ipykernel', types.ModuleType('ipykernel'))
    assert fork_unsafe_in_notebook('fork') is True


def test_darwin_shell_with_kernel_is_unsafe(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('OtherShell', kernel=object()))
    assert fork_unsafe_in_notebook('fork') is True


# ── macOS never forks (S0 of the parallelization plan) ─────────────────────

def test_fork_unsafe_on_darwin_everywhere(monkeypatch):
    """No kernel, no IPython: a plain macOS script still never forks."""
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython', None)
    assert fork_unsafe('fork') is True
    assert fork_unsafe('spawn') is False
    assert fork_unsafe('forkserver') is False


def test_fork_allowed_on_linux_even_in_a_kernel(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setitem(sys.modules, 'ipykernel', types.ModuleType('ipykernel'))
    assert fork_unsafe('fork') is False


@pytest.mark.parametrize('name,bases,attrs,ipykernel', [
    ('ZMQInteractiveShell', (), {}, True),                    # Jupyter / VS Code
    ('SageZMQInteractiveShell', ('ZMQInteractiveShell',), {}, False),   # SageMath
    ('SpyderShell', ('ZMQInteractiveShell',), {}, False),     # Spyder
    ('VSCodeShell', (), {}, True),                            # any ipykernel
    ('OtherShell', (), {'kernel': object()}, False),          # shell with a kernel
])
def test_fake_kernel_shells_are_detected(monkeypatch, name, bases, attrs,
                                          ipykernel):
    base = tuple(type(b, (), {}) for b in bases)
    monkeypatch.setitem(sys.modules, 'IPython', _fake_ipython(name, base, **attrs))
    if ipykernel:
        monkeypatch.setitem(sys.modules, 'ipykernel',
                            types.ModuleType('ipykernel'))
    assert in_jupyter_kernel() is True
    monkeypatch.setattr(sys, 'platform', 'darwin')
    assert fork_unsafe_in_notebook('fork') is True and fork_unsafe('fork')


def test_terminal_shell_is_not_a_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('TerminalInteractiveShell'))
    assert in_jupyter_kernel() is False


# ── DAEDALUS_PARALLEL kill switch ──────────────────────────────────────────

@pytest.mark.parametrize('raw,off', [
    (None, False), ('', False), ('1', False), ('true', False), (' ON ', False),
    ('0', True), ('false', True), ('No', True), ('off', True)])
def test_parallel_disabled_values(raw, off):
    env = {} if raw is None else {fs.PARALLEL_ENV: raw}
    assert parallel_disabled(env) is off


def test_parallel_disabled_rejects_a_typo():
    with pytest.raises(ValueError, match='DAEDALUS_PARALLEL'):
        parallel_disabled({fs.PARALLEL_ENV: 'fasle'})


def test_parallel_disabled_reads_the_environment_at_call_time(monkeypatch):
    monkeypatch.setenv(fs.PARALLEL_ENV, '0')
    assert parallel_disabled() is True
    monkeypatch.delenv(fs.PARALLEL_ENV)
    assert parallel_disabled() is False


def test_fallback_warns_once_per_process(monkeypatch):
    monkeypatch.setattr(fs, '_WARNED', set())
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fs.warn_fork_fallback_once('site A')
        fs.warn_fork_fallback_once('site B')
        fs.warn_fork_fallback_once('site A')
    msgs = [str(r.message) for r in rec]
    assert len(msgs) == 1 and msgs[0].startswith('site A')
    assert 'disabled on macOS' in msgs[0] and 'DAEDALUS_PARALLEL=0' in msgs[0]
    assert rec[0].category is RuntimeWarning


# ── OBJC_DISABLE_INITIALIZE_FORK_SAFETY is never set ───────────────────────

_SETS_OBJC = re.compile(
    r"(setdefault|putenv|environ\s*\[)\s*\(?\s*['\"]OBJC_DISABLE_INITIALIZE_FORK_SAFETY")


def test_package_never_sets_the_objc_fork_safety_flag():
    """The runtime reads it once at image load, so setting it from Python never
    guarded the forks it was meant for; it only leaked into every process
    exec'd later, where it turns the runtime's controlled abort into a hard
    crash (the June 2026 machine crashes)."""
    hits = []
    for sub in ('engine', 'api', 'simulations', 'tests', 'daedalus.py'):
        base = ROOT / sub
        files = [base] if base.is_file() else sorted(base.rglob('*.py'))
        for f in files:
            if f.resolve() == Path(__file__).resolve():
                continue
            if _SETS_OBJC.search(f.read_text(errors='replace')):
                hits.append(str(f.relative_to(ROOT)))
    assert not hits, f'sets OBJC_DISABLE_INITIALIZE_FORK_SAFETY: {hits}'
    assert _SETS_OBJC.search("os.environ.setdefault(\n    'OBJC_DISABLE_"
                             "INITIALIZE_FORK_SAFETY', 'YES')")


# ── Linux only: ECL (Maxima) garbage collection in a forked child ──────────

def _ecl_gc_in_child():
    from sage.all import cos, sin, var
    from sage.libs.ecl import ecl_eval
    x = var('x')
    r = (sin(x) ** 2 + cos(x) ** 2).simplify_full()      # Maxima, via ECL
    ecl_eval('(si::gc t)')                               # full collection
    return str(r)


@pytest.mark.skipif(sys.platform == 'darwin', reason='macOS never forks')
def test_ecl_gc_in_a_forked_child_completes():
    """A Sage parent runs Boehm GC marker threads (libgc, ECL) after its first
    symbolic step; a fork child has the GC state but not the threads.  Type
    assignment forks such a parent on Linux, so a child that simplifies and
    collects must finish (a hang would show up here as a timeout, not as a
    stuck suite)."""
    import multiprocessing as mp
    from sage.all import cos, sin, var
    from sage.libs.ecl import ecl_eval
    x = var('x')
    assert str((sin(x) ** 2 + cos(x) ** 2).simplify_full()) == '1'
    ecl_eval('(si::gc t)')
    pool = mp.get_context('fork').Pool(1)
    try:
        assert pool.apply_async(_ecl_gc_in_child).get(timeout=120) == '1'
    finally:
        pool.terminate()
        pool.join()


def test_spatial_threads_gate_honours_the_kill_switch(monkeypatch):
    from engine.integration.spatial.pipeline_bridge import _spatial_threads_gate
    monkeypatch.delenv(fs.PARALLEL_ENV, raising=False)
    assert _spatial_threads_gate(True, True, 4, 100) is True
    assert _spatial_threads_gate(True, False, 4, 100) is False     # L=1 only
    assert _spatial_threads_gate(False, True, 4, 100) is False
    monkeypatch.setenv(fs.PARALLEL_ENV, '0')
    assert _spatial_threads_gate(True, True, 4, 100) is False


def test_fork_unsafe_where_the_platform_has_no_fork(monkeypatch):
    import multiprocessing as mp
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(mp, 'get_all_start_methods', lambda: ['spawn'])
    assert fork_unsafe('fork') is True and fork_unsafe('spawn') is False

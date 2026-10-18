"""Unit tests for the shared fork-safety guard (engine/fork_safety.py).

Forking after Cocoa/BLAS init inside a macOS Jupyter kernel can hard-crash the
kernel AND the OS, so every fork-based path must consult
``fork_unsafe_in_notebook`` and degrade to serial in exactly that case — and
only that case (fork must stay available for scripts / pytest / Linux /
terminal IPython, where it is safe and fast).

Run:  sage -python -m pytest tests/test_fork_safety.py -q
"""
import sys
import types

from engine.fork_safety import fork_unsafe_in_notebook


import pytest


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


def test_darwin_terminal_ipython_is_safe(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython',
                        _fake_ipython('TerminalInteractiveShell'))
    assert fork_unsafe_in_notebook('fork') is False


def test_darwin_no_ipython_is_safe(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'IPython', None)   # import → fails → False
    assert fork_unsafe_in_notebook('fork') is False


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

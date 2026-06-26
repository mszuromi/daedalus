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


def _fake_ipython(shell_class_name):
    """Install a fake ``IPython`` module whose get_ipython() returns a shell
    whose class is named ``shell_class_name``."""
    shell = type(shell_class_name, (), {})()
    mod = types.SimpleNamespace(get_ipython=lambda: shell)
    return mod


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

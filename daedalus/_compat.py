"""daedalus._compat -- the old top-level import names, deprecated.

The library used to be three top-level packages, ``api``, ``engine`` and
``simulations``; they now live under ``daedalus``.  For one release the old
names keep working: ``import engine.core.vertices`` gives the module
``daedalus.engine.core.vertices`` itself (the same object, so its state,
classes and exceptions are shared, and an old pickle that names
``engine.<...>`` classes still loads), with one ``DeprecationWarning`` per
old name.

:func:`install` puts an alias finder at the front of ``sys.meta_path`` (the
front, so a submodule of an alias never loads a second copy from the
package's ``__path__``).  It claims a top-level old name only when no other
``api`` / ``engine`` / ``simulations`` package is importable, so a user's
own package of that name always wins, and a submodule only while its
top-level name is our alias.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings

#: The old top-level package names.
OLD_NAMES = ('api', 'engine', 'simulations')

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT = os.path.dirname(_PACKAGE_DIR)
_warned = set()


def _is_our_shim(spec) -> bool:
    """A spec for the checkout's own forwarding package (``<repo>/api/``)."""
    origin = getattr(spec, 'origin', None) or ''
    return os.path.dirname(os.path.dirname(os.path.abspath(origin))) == _CHECKOUT


def _ours(filename: str) -> bool:
    """A frame of this module or of a checkout's forwarding package."""
    return filename == __file__ or any(
        filename == os.path.join(_CHECKOUT, n, '__init__.py') for n in OLD_NAMES)


def _internal(filename: str) -> bool:
    """A frame ``warnings`` itself skips when it counts ``stacklevel``."""
    return 'importlib' in filename and '_bootstrap' in filename


def warn_old_name(top: str) -> None:
    """One ``DeprecationWarning`` per old name, attributed to the importer."""
    if top in _warned:
        return
    _warned.add(top)
    f, level = sys._getframe(1), 2
    while f is not None and (_internal(f.f_code.co_filename)
                             or _ours(f.f_code.co_filename)):
        if not _internal(f.f_code.co_filename):
            level += 1
        f = f.f_back
    warnings.warn(f'`{top}` is the old top-level name of `daedalus.{top}`; import '
                  f'`daedalus.{top}` instead (the old name is removed in the next '
                  f'release)', DeprecationWarning, stacklevel=level)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class AliasFinder(importlib.abc.MetaPathFinder):
    """``api[.x]`` / ``engine[.x]`` / ``simulations[.x]`` -> ``daedalus.<...>``."""

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition('.')[0]
        if top not in OLD_NAMES:
            return None
        if fullname == top:
            if top in sys.modules:
                return None
            other = importlib.machinery.PathFinder.find_spec(top)
            if other is not None and not _is_our_shim(other):
                return None
            warn_old_name(top)
        else:
            alias = sys.modules.get(top)
            if alias is None or alias is not sys.modules.get(f'daedalus.{top}'):
                return None
        target = f'daedalus.{fullname}'
        try:
            target_spec = importlib.util.find_spec(target)
        except ModuleNotFoundError:
            return None
        if target_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(target),
            is_package=target_spec.submodule_search_locations is not None)


_FINDER = AliasFinder()


def install() -> None:
    """Put the alias finder first on ``sys.meta_path`` (idempotent)."""
    if not any(isinstance(f, AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)


def forward(name: str) -> None:
    """Body of a checkout's forwarding package ``<repo>/<name>/__init__.py``:
    replace it with ``daedalus.<name>`` in ``sys.modules``."""
    install()
    warn_old_name(name)
    sys.modules[name] = importlib.import_module(f'daedalus.{name}')

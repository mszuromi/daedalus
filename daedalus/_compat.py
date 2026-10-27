"""daedalus._compat -- the old top-level import names, deprecated.

The library used to be three top-level packages, ``api``, ``engine`` and
``simulations``; they now live under ``daedalus``.  For one release the old
names keep working: ``import engine.core.vertices`` gives the module
``daedalus.engine.core.vertices`` itself (the same object, so its state,
classes and exceptions are shared, and an old pickle that names
``engine.<...>`` classes still loads once ``daedalus`` is imported), with
one ``DeprecationWarning`` per old name.

:func:`install` puts an alias finder at the front of ``sys.meta_path`` (the
front, so a submodule of an alias never loads a second copy from the
package's ``__path__``).  It claims a top-level old name unless a user's own
regular package of that name is importable (that one always wins), and a
submodule only while its top-level name is our alias.  The aliased module
keeps its own ``__spec__``, so ``importlib.reload`` and autoreload still work
on it.

In a source checkout, a module that exists only in the checkout's old
top-level folder (``<repo>/simulations/my_sim.py``, a local file git does
not track) still imports under its old name, from that folder.
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


def _in_checkout() -> bool:
    return os.path.isfile(os.path.join(_CHECKOUT, 'pyproject.toml'))


def _shim_init(name: str) -> str:
    return os.path.join(_CHECKOUT, name, '__init__.py')


def _yields_to(spec, top: str) -> bool:
    """Whether the finder leaves ``top`` to ``spec`` (what ``sys.path``
    would import): only a user's own regular package or module does.  Not
    the checkout's forwarding package, not a bare directory (a namespace
    package, e.g. a results folder named ``engine``), and not a copy of our
    own subpackage reached through a path that points inside ``daedalus/``
    (that would be a second copy of the library)."""
    if spec is None or spec.origin is None or spec.origin == 'namespace':
        return False
    origin = os.path.abspath(spec.origin)
    if _in_checkout() and origin == _shim_init(top):
        return False
    if origin.startswith(_PACKAGE_DIR + os.sep):
        return False
    return True


def _ours(filename: str) -> bool:
    """A frame of this module or of a checkout's forwarding package."""
    return filename == __file__ or any(filename == _shim_init(n) for n in OLD_NAMES)


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
    """Hands out the already-imported ``daedalus.<...>`` module."""

    def __init__(self, target):
        self.target = target
        self._spec = None

    def create_module(self, spec):
        if '.' not in spec.name:
            warn_old_name(spec.name)
        module = importlib.import_module(self.target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        # the import system set the alias spec on the module: put the
        # module's own back (reload, autoreload and pickling use it)
        module.__spec__ = self._spec

    def get_code(self, fullname):
        """For ``python -m <old name>``: the target module's code."""
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)


class AliasFinder(importlib.abc.MetaPathFinder):
    """``api[.x]`` / ``engine[.x]`` / ``simulations[.x]`` -> ``daedalus.<...>``."""

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition('.')[0]
        if top not in OLD_NAMES:
            return None
        if fullname == top:
            if top in sys.modules:
                return None
            if _yields_to(importlib.machinery.PathFinder.find_spec(top), top):
                return None
        else:
            alias = sys.modules.get(top)
            if alias is None or alias is not sys.modules.get(f'daedalus.{top}'):
                return None
        target_name = f'daedalus.{fullname}'
        try:
            target_spec = importlib.util.find_spec(target_name)
        except ModuleNotFoundError:
            target_spec = None
        if target_spec is None:
            return self._checkout_local(fullname)
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(target_name),
            is_package=target_spec.submodule_search_locations is not None)

    @staticmethod
    def _checkout_local(fullname):
        """A module only the checkout's old folder holds (an untracked local
        file), imported from there under its old name."""
        if not _in_checkout():
            return None
        parts = fullname.split('.')
        where = os.path.join(_CHECKOUT, *parts[:-1])
        if not os.path.isdir(where):
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [where])


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

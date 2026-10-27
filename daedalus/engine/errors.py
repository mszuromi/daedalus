"""The common base of Daedalus's own exceptions.

``except daedalus.DaedalusError`` (also ``api.DaedalusError``) catches every
refusal and failure the package raises on purpose -- a model it cannot
handle, an option it rejects, a resource budget it enforces -- without
catching a bug (``KeyError``, ``AttributeError`` ...) or an error from a
library.

Every exception class the package defined before this base existed is kept
and is now a subclass of it AND of the builtin it always was (``ValueError``,
``RuntimeError``, ``NotImplementedError``, ``MemoryError`` ...), so existing
``except ValueError`` / ``except NotImplementedError`` clauses keep working.
The builtins the package still raises directly (a bare ``ValueError`` for a
bad argument) are unchanged.
"""
from __future__ import annotations


class DaedalusError(Exception):
    """Base of the exceptions Daedalus raises on purpose."""

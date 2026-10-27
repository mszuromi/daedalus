"""
engine._env
===========
One parser for the package's environment switches (the
values were parsed seven different ways at 32 sites, garbage silently meant
"on" at some and "off" at others, and the two memory budgets gave ``0``
opposite meanings).

Every reader takes ``environ`` (default ``os.environ``) and raises
``ValueError`` naming the variable on a value it does not understand; an
unset or empty variable is the default.

* :func:`flag` -- on/off: ``1 true yes on`` / ``0 false no off``.
* :func:`choice` -- one of a fixed set of words (case-insensitive).
* :func:`integer`, :func:`number` -- with an optional lower bound.
* :func:`budget_gb` -- the memory budgets (``PHASE_J_MEM_BUDGET_GB``,
  ``SPATIAL_MEM_BUDGET_GB``), one semantics for both: a positive number of
  GB; ``off`` / ``none`` / ``inf`` / ``unlimited`` = no limit; ``0`` or a
  negative value raises (it meant "no limit" for one guard and "refuse
  everything" for the other).
* :func:`retired` -- a retired switch that is set: one ``FutureWarning``
  per process and name, the value is ignored.
"""
import math
import os
import warnings

_TRUE = ('1', 'true', 'yes', 'on')
_FALSE = ('0', 'false', 'no', 'off')
_NO_LIMIT = ('off', 'none', 'inf', 'infinity', 'unlimited')


def _raw(name, environ):
    env = os.environ if environ is None else environ
    v = env.get(name)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def flag(name, default, environ=None):
    """``True`` / ``False`` from ``name``; ``default`` when unset or empty."""
    v = _raw(name, environ)
    if v is None:
        return bool(default)
    low = v.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f'{name}={v!r}: expected an on/off value '
                     f'({"/".join(_TRUE)} or {"/".join(_FALSE)})')


def choice(name, choices, default, environ=None):
    """The lower-cased value of ``name`` if it is one of ``choices``;
    ``default`` when unset or empty."""
    v = _raw(name, environ)
    if v is None:
        return default
    low = v.lower()
    if low not in choices:
        raise ValueError(f'{name}={v!r}: expected one of '
                         f'{", ".join(repr(c) for c in choices)}')
    return low


def integer(name, default, minimum=None, environ=None):
    """``int`` value of ``name`` (at least ``minimum`` when given)."""
    v = _raw(name, environ)
    if v is None:
        return default
    try:
        x = int(v)
    except ValueError:
        raise ValueError(f'{name}={v!r}: expected an integer') from None
    if minimum is not None and x < minimum:
        raise ValueError(f'{name}={v!r}: expected an integer >= {minimum}')
    return x


def number(name, default, minimum=None, positive=False, environ=None):
    """Finite ``float`` value of ``name`` (``> 0`` with ``positive``; at
    least ``minimum`` when given)."""
    v = _raw(name, environ)
    if v is None:
        return default
    try:
        x = float(v)
    except ValueError:
        raise ValueError(f'{name}={v!r}: expected a number') from None
    if not math.isfinite(x):
        raise ValueError(f'{name}={v!r}: expected a finite number')
    if positive and not x > 0:
        raise ValueError(f'{name}={v!r}: expected a positive number')
    if minimum is not None and x < minimum:
        raise ValueError(f'{name}={v!r}: expected a number >= {minimum:g}')
    return x


def budget_gb(name, environ=None):
    """A memory budget: ``None`` when unset or empty (the caller's default),
    ``math.inf`` for no limit (``off`` / ``none`` / ``inf`` /
    ``unlimited``), else the positive number of GB.  ``0`` and negative
    values raise."""
    v = _raw(name, environ)
    if v is None:
        return None
    if v.lower() in _NO_LIMIT:
        return math.inf
    try:
        gb = float(v)
    except ValueError:
        raise ValueError(f'{name}={v!r} is not a number (GB), or off') \
            from None
    if gb == math.inf:
        return math.inf
    if not gb > 0:
        raise ValueError(f'{name}={v!r}: a memory budget must be a positive '
                         f'number of GB (use {name}=off for no limit)')
    return gb


_RETIRED_SEEN = set()


def retired(name, note, environ=None):
    """Warn once per process (``FutureWarning``) when the retired switch
    ``name`` is set; its value is ignored.  ``note``: what replaced it.
    Returns whether it was set."""
    if _raw(name, environ) is None:
        return False
    if name not in _RETIRED_SEEN:
        _RETIRED_SEEN.add(name)
        from engine._warn_helpers import ensure_warning_shown
        msg = f'{name} is retired and ignored: {note}'
        ensure_warning_shown(FutureWarning, message=msg)
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return True


#: ``DAEDALUS_PHASE_J_ENGINE``: the Phase J engine (the ONE reader; there used to be five, four of them unvalidated).
PHASE_J_ENGINE_ENV = 'DAEDALUS_PHASE_J_ENGINE'
_ENGINE_WORDS = {'': 'statespace', 'default': 'statespace',
                 'statespace': 'statespace', 'poles': 'poles'}


def phase_j_engine(environ=None):
    """``'statespace'`` (unset, ``default``, ``statespace``) or ``'poles'``;
    any other value raises."""
    env = os.environ if environ is None else environ
    raw = str(env.get(PHASE_J_ENGINE_ENV, '')).strip().lower()
    try:
        return _ENGINE_WORDS[raw]
    except KeyError:
        raise ValueError(f'{PHASE_J_ENGINE_ENV}={raw!r}: expected statespace '
                         f'(the default), default or poles (or unset)') \
            from None

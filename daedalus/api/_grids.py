"""One rule for reading a grid option (``tau_grid``, ``chi_grid`` /
``spatial_grid``), shared by ``compute_cumulants``, ``dd.Config`` and the YAML
config -- they used to read a 3-entry sequence three different ways.

* an ``(lo, hi, n)`` TUPLE is a ``linspace`` RANGE of ``n`` points;
* a list or array is a list of LITERAL points;
* a 3-element LIST whose last entry is an integer count >= 2
  (``[-5, 5, 51]``) is the classic "I meant a range" slip -- read literally it
  would be the three points {-5, 5, 51} -- and is refused with a message that
  names both spellings;
* ``None`` stays ``None`` (use the default grid).
"""
from __future__ import annotations

import numpy as np


def is_real_number(v):
    """``v`` is a real scalar: a Python / numpy number or a Sage ``Integer`` /
    ``Rational`` / ``RealNumber`` (the Sage Jupyter kernel turns every literal
    into one of those, and none is an ``int`` / ``float``).  Not a bool, not a
    string."""
    if isinstance(v, (bool, np.bool_, str, bytes, complex, np.complexfloating)):
        return False
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def is_integer_number(v):
    """``v`` is a real scalar with an integer value (a Sage ``Integer``, or
    ``13.0``)."""
    if not is_real_number(v):
        return False
    try:
        return float(v) == int(v)
    except (OverflowError, ValueError):
        return False


def require_finite(arr, name):
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(
            f"{name} must be finite: {int(bad.sum())} of {arr.size} "
            f"entries are NaN or infinite.")


def resolve_grid(g, name='grid'):
    """The grid ``g`` as a float array under the rule in the module
    docstring, or ``None`` for ``None``."""
    if g is None:
        return None
    if isinstance(g, tuple):
        if len(g) != 3:
            raise ValueError(
                f"{name} tuple must be (lo, hi, n); got a {len(g)}-tuple "
                f"{g!r}. For an arbitrary grid pass a list/array instead.")
        n = g[2]
        if not is_integer_number(n) or int(n) < 1:
            raise ValueError(
                f"{name}={g!r}: the third entry of a (lo, hi, n) range is "
                f"the number of points and must be an integer >= 1.")
        arr = np.linspace(g[0], g[1], int(n))
        require_finite(arr, name)
        return arr
    if (isinstance(g, list) and len(g) == 3
            and is_integer_number(g[2]) and float(g[2]) >= 2):
        lo, hi, n = g[0], g[1], int(g[2])
        raise ValueError(
            f"{name}={g!r} is a 3-element LIST, so it is read as 3 literal "
            f"grid points {{{float(lo)}, {float(hi)}, {float(n)}}} — note the "
            f"{n}!  For a linspace RANGE of {n} points from {lo} to {hi}, use "
            f"a TUPLE: {name}=({lo}, {hi}, {n}).  For {n} literal points, pass "
            f"a numpy array: {name}=np.array({g!r}).")
    arr = np.asarray(g, dtype=float)
    if arr.size == 0:
        raise ValueError(
            f"{name} is empty — pass a non-empty grid, or omit it (None) to use "
            f"the default.")
    require_finite(arr, name)
    return arr

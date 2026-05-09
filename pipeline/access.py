"""
pipeline.access — natural-name accessors for compute_cumulants results.

Hides the MSR-JD internal naming (``dn`` for fluctuations, ``nstar``
for saddle values, etc.) behind a single user-facing convention:

  - **Field base names** are the physical letters: ``'n'`` for the
    spike train, ``'v'`` for voltage, ``'m'`` for an external rate.
  - **Fluctuation fields** prepend ``d``: ``'dn'``, ``'dv'``, ``'dm'``.
    These are what the action expansion uses internally.
  - **Mean-field saddle values** append ``star``: ``'nstar'``,
    ``'vstar'``, ``'mstar'``.

Three things are exposed:

  - :class:`MeanField` — ``mf['v', 1]`` returns ``v*_1``, ``mf['n']``
    returns the whole nstar vector.
  - :class:`Parameters` — ``params['E', 1]``, ``params['w', 1, 2]``,
    ``params['tau']``.
  - :func:`normalize_external_fields` — translates user-facing
    ``[('n', 1), ('n', 2)]`` to internal ``[('dn', 1), ('dn', 2)]``.
    Already-internal names pass through unchanged for back-compat.

Convention is hard-coded (``'n' → 'dn'``, ``'n' → 'nstar'``).  This
covers the 5 standardized theory files; future models that use
different physical letters can extend ``_BASE_TO_FLUCT`` /
``_BASE_TO_MF``.
"""
from __future__ import annotations

from typing import Any


# Convention map — base physical letter → internal naming.
_BASE_TO_FLUCT = {
    'n': 'dn',  'dn': 'dn',
    'v': 'dv',  'dv': 'dv',
    'm': 'dm',  'dm': 'dm',
}

_BASE_TO_MF = {
    'n': 'nstar',  'nstar': 'nstar',
    'v': 'vstar',  'vstar': 'vstar',
    'm': 'mstar',  'mstar': 'mstar',
}


def normalize_external_fields(external_fields):
    """Translate user-facing ``[('n', 1), ('v', 2)]`` to internal
    fluctuation names ``[('dn', 1), ('dv', 2)]``.

    Already-internal names pass through unchanged so old code keeps
    working::

        >>> normalize_external_fields([('dn', 1)])
        [('dn', 1)]
        >>> normalize_external_fields([('n', 1), ('v', 2)])
        [('dn', 1), ('dv', 2)]
    """
    out = []
    for name, pop in external_fields:
        canonical = _BASE_TO_FLUCT.get(name, name)
        out.append((canonical, int(pop)))
    return out


class MeanField:
    """Read-only accessor for the saddle-point values returned by
    ``solve_mean_field``.

    Indexing
    --------
    Tuple indexing with 1-based population numbers::

        mf['n', 1]   →  n*_1   (first population's mean firing rate)
        mf['v', 2]   →  v*_2   (second population's voltage)
        mf['m', 1]   →  m*_1   (only when the model declares mstar)

    Bare-string indexing returns the whole length-npop vector::

        mf['n']      →  [n*_1, n*_2, ...]
        mf['v']      →  [v*_1, v*_2, ...]

    Both natural and internal names work — ``mf['v', 1]`` and
    ``mf['vstar', 1]`` are equivalent.

    Iteration / membership
    ----------------------
    ``for name in mf`` yields the natural keys present, e.g.
    ``['n', 'v']`` for a Hawkes model and ``['n', 'v', 'm']`` for GTaS.
    ``'v' in mf`` works.

    Plain-dict view
    ---------------
    ``mf.as_dict()`` returns ``{name: list_of_values}`` keyed by
    natural name — handy for templating into prints or saves.
    """

    # Reverse lookup: internal → natural.  Take the first base entry
    # for each internal so 'nstar' → 'n' (not 'nstar' → 'nstar').
    _MF_TO_BASE = {'nstar': 'n', 'vstar': 'v', 'mstar': 'm'}

    def __init__(self, mf_values: dict[str, list]):
        # Source of truth: the adaptive {internal_name: [v1, v2, ...]}
        # dict that compute_cumulants builds.  Keys may already be
        # internal ('nstar') or natural ('n') depending on caller —
        # we normalize to internal for storage.
        self._by_internal: dict[str, list[float]] = {}
        for k, v in (mf_values or {}).items():
            internal = _BASE_TO_MF.get(k, k)
            self._by_internal[internal] = list(v) if v is not None else None

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, pop = key[0], int(key[1])
            internal = _BASE_TO_MF.get(name, name)
            vals = self._by_internal.get(internal)
            if vals is None:
                raise KeyError(f'unknown mean-field key: {name!r}')
            return float(vals[pop - 1])

        # Bare string → whole vector
        internal = _BASE_TO_MF.get(key, key)
        vals = self._by_internal.get(internal)
        if vals is None:
            raise KeyError(f'unknown mean-field key: {key!r}')
        return list(vals)

    def __contains__(self, key):
        if isinstance(key, tuple):
            name = key[0]
        else:
            name = key
        internal = _BASE_TO_MF.get(name, name)
        return internal in self._by_internal and \
               self._by_internal[internal] is not None

    def __iter__(self):
        for internal in self._by_internal:
            if self._by_internal[internal] is None:
                continue
            yield self._MF_TO_BASE.get(internal, internal)

    def keys(self):
        return list(iter(self))

    def as_dict(self) -> dict[str, list[float]]:
        """Plain ``{natural_name: [v1, v2, ...]}`` dict view."""
        return {self._MF_TO_BASE.get(k, k): list(v)
                for k, v in self._by_internal.items()
                if v is not None}

    def __repr__(self):
        items = ', '.join(f'{k}={v}' for k, v in self.as_dict().items())
        return f'MeanField({items})'


class Parameters:
    """Read-only accessor for fundamental model parameters.

    Indexing matches the user's input shape:

      - Scalars (e.g. ``tau``)::                ``params['tau']``
      - Vectors (e.g. ``E``)::                  ``params['E', 1]``
      - Matrices (e.g. ``w``)::                 ``params['w', 1, 2]``

    All array indices are 1-based to match population numbering.
    Bare-string indexing returns the raw value::

        params['E']    →  [0.78, 0.81]
        params['w']    →  [[0.30, 0.25], [0.30, 0.35]]
    """

    def __init__(self, fundamental: dict):
        self._raw = dict(fundamental or {})

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name = key[0]
            indices = key[1:]
            val = self._raw[name]
            for i in indices:
                val = val[int(i) - 1]      # 1-based
            return val
        return self._raw[key]

    def __contains__(self, key):
        if isinstance(key, tuple):
            return key[0] in self._raw
        return key in self._raw

    def __iter__(self):
        return iter(self._raw)

    def keys(self):
        return list(self._raw.keys())

    def values(self):
        return list(self._raw.values())

    def items(self):
        return list(self._raw.items())

    def as_dict(self) -> dict:
        """Plain ``{name: value}`` dict view (copy)."""
        return dict(self._raw)

    def __repr__(self):
        keys = ', '.join(self._raw.keys())
        return f'Parameters({keys})'

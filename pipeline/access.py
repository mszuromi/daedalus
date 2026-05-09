"""
pipeline.access — natural-name accessors for compute_cumulants results.

Hides the MSR-JD internal naming (``dn`` for fluctuations, ``nstar``
for saddle values, etc.) behind the model's declared user-facing
naming convention.

Source of truth
---------------
The mapping comes from ``model['naming_convention']``, which is built
by ``TheoryBuilder.build()`` from the ``natural_name=`` declarations
on physical fields and parameters::

    .physical_field('dn', natural_name='n', ...)
    .parameter('nstar', mean_field=True, natural_name='n', ...)

This produces::

    naming_convention = {
        'fluctuation_fields': {'n': 'dn', ...},     # natural → internal
        'mean_field_saddles': {'n': 'nstar', ...},  # natural → internal
        'mf_parameters': ['nstar', 'vstar', ...],   # internal names
    }

Models that don't declare a naming convention fall back to the
classic Hawkes ``n``/``v``/``m`` map below — so existing model files
without ``natural_name=`` annotations keep working.

User-facing primitives
----------------------
- :class:`MeanField` — ``mf['v', 1]`` returns ``v*_1``, ``mf['n']``
  returns the whole nstar vector.
- :class:`Parameters` — ``params['E', 1]``, ``params['w', 1, 2]``,
  ``params['tau']``.
- :func:`normalize_external_fields` — translates user-facing
  ``[('n', 1), ('n', 2)]`` to internal ``[('dn', 1), ('dn', 2)]``.
"""
from __future__ import annotations

from typing import Optional


# ── Default fallback naming map ───────────────────────────────────────
# Used when a model dict has no ``naming_convention`` (i.e. older
# hand-written model files in models/).  Covers the classic Hawkes
# letters n/v/m.  Extend this only if you want to bake new defaults
# into the pipeline itself; the preferred path for new physics is to
# declare ``natural_name=`` in the TheoryBuilder.

_DEFAULT_FLUCT = {
    'n':  'dn',  'dn': 'dn',
    'v':  'dv',  'dv': 'dv',
    'm':  'dm',  'dm': 'dm',
}

_DEFAULT_SADDLE = {
    'n': 'nstar',  'nstar': 'nstar',
    'v': 'vstar',  'vstar': 'vstar',
    'm': 'mstar',  'mstar': 'mstar',
}


def _build_fluct_map(naming_convention: Optional[dict]) -> dict[str, str]:
    """Compose the natural→internal fluctuation map.

    Order of precedence (later wins):
      1. ``_DEFAULT_FLUCT`` (n/v/m fallback)
      2. ``naming_convention['fluctuation_fields']`` (model declarations)
      3. internal-name passthrough (each declared internal name maps to
         itself, so users can pass either form)
    """
    out = dict(_DEFAULT_FLUCT)
    if naming_convention:
        for natural, internal in (naming_convention.get(
                'fluctuation_fields') or {}).items():
            out[natural]  = internal
            out[internal] = internal
    return out


def _build_saddle_map(naming_convention: Optional[dict]) -> dict[str, str]:
    """Compose the natural→internal mean-field saddle map.

    Same precedence story as ``_build_fluct_map``.
    """
    out = dict(_DEFAULT_SADDLE)
    if naming_convention:
        for natural, internal in (naming_convention.get(
                'mean_field_saddles') or {}).items():
            out[natural]  = internal
            out[internal] = internal
    return out


def _build_reverse_saddle_map(saddle_map: dict[str, str]) -> dict[str, str]:
    """Reverse map for display: internal → natural.

    Picks the SHORTEST natural name pointing at each internal value
    (so 'nstar' → 'n' and not 'nstar' → 'nstar').
    """
    rev: dict[str, str] = {}
    for natural, internal in saddle_map.items():
        if internal in rev and len(rev[internal]) <= len(natural):
            continue
        rev[internal] = natural
    return rev


def normalize_external_fields(external_fields,
                              naming_convention: Optional[dict] = None):
    """Translate user-facing external fields to internal fluctuation
    names::

        normalize_external_fields([('n', 1)])              → [('dn', 1)]
        normalize_external_fields([('r', 1)],              # custom letter
            naming_convention={'fluctuation_fields': {'r': 'dr'}})
                                                            → [('dr', 1)]

    Already-internal names pass through unchanged.  When
    ``naming_convention`` is omitted, the n/v/m fallback applies.
    """
    fluct_map = _build_fluct_map(naming_convention)
    out = []
    for name, pop in external_fields:
        canonical = fluct_map.get(name, name)
        out.append((canonical, int(pop)))
    return out


class MeanField:
    """Read-only accessor for the saddle-point values returned by
    ``solve_mean_field``.

    Indexing
    --------
    Tuple indexing with 1-based population numbers::

        mf['n', 1]   →  n*_1   (first population's saddle for natural 'n')
        mf['v', 2]   →  v*_2

    Bare-string indexing returns the whole length-npop vector::

        mf['n']      →  [n*_1, n*_2, ...]

    Both natural and internal names work — ``mf['v', 1]`` and
    ``mf['vstar', 1]`` are equivalent (when 'v' is declared as
    natural_name for vstar).

    Construction
    ------------
    ``MeanField(mf_values, naming_convention=...)`` consumes the
    adaptive ``{internal_name: [v1, v2, ...]}`` dict that
    ``compute_cumulants`` builds, plus the model's naming convention.
    """

    def __init__(self, mf_values: dict[str, list],
                 naming_convention: Optional[dict] = None):
        self._saddle_map = _build_saddle_map(naming_convention)
        self._reverse    = _build_reverse_saddle_map(self._saddle_map)

        # Normalize keys to internal form for storage.
        self._by_internal: dict[str, list[float]] = {}
        for k, v in (mf_values or {}).items():
            internal = self._saddle_map.get(k, k)
            self._by_internal[internal] = (
                list(v) if v is not None else None)

    def _resolve(self, name: str) -> str:
        """Translate either natural or internal name to internal."""
        return self._saddle_map.get(name, name)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, pop = key[0], int(key[1])
            internal = self._resolve(name)
            vals = self._by_internal.get(internal)
            if vals is None:
                raise KeyError(f'unknown mean-field key: {name!r}')
            return float(vals[pop - 1])

        internal = self._resolve(key)
        vals = self._by_internal.get(internal)
        if vals is None:
            raise KeyError(f'unknown mean-field key: {key!r}')
        return list(vals)

    def __contains__(self, key):
        name = key[0] if isinstance(key, tuple) else key
        internal = self._resolve(name)
        return (internal in self._by_internal
                and self._by_internal[internal] is not None)

    def __iter__(self):
        for internal in self._by_internal:
            if self._by_internal[internal] is None:
                continue
            yield self._reverse.get(internal, internal)

    def keys(self):
        return list(iter(self))

    def as_dict(self) -> dict[str, list[float]]:
        """Plain ``{natural_name: [v1, v2, ...]}`` dict view."""
        return {self._reverse.get(k, k): list(v)
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
    Bare-string indexing returns the raw value.
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

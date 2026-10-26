"""
api.access — natural-name accessors for compute_cumulants results.

Hides the MSR-JD internal naming (``dn`` for fluctuations, ``nstar``
for saddle values, etc.) behind the model's declared user-facing
naming convention.

Source of truth
---------------
The mapping comes from ``model['naming_convention']``, which is built
by ``ModelBuilder.build()`` from the ``natural_name=`` declarations
on physical fields and parameters::

    .physical_field('dn', natural_name='n', ...)
    .parameter('nstar', mean_field=True, natural_name='n', ...)

This produces::

    naming_convention = {
        'fluctuation_fields': {'n': 'dn', ...},     # natural → internal
        'mean_field_saddles': {'n': 'nstar', ...},  # natural → internal
        'mf_parameters': ['nstar', 'vstar', ...],   # internal names
    }

Without a naming convention only the internal names are accepted (no
built-in letter table: every ModelBuilder model declares its convention).

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


def _build_fluct_map(naming_convention: Optional[dict]) -> dict[str, str]:
    """Compose the natural→internal fluctuation map.

    ``naming_convention['fluctuation_fields']`` (the model's declarations),
    each declared internal name also mapping to itself, so users can pass
    either form.
    """
    out: dict[str, str] = {}
    if naming_convention:
        for natural, internal in (naming_convention.get(
                'fluctuation_fields') or {}).items():
            out[natural]  = internal
            out[internal] = internal
    return out


def _build_saddle_map(naming_convention: Optional[dict]) -> dict[str, str]:
    """Compose the natural→internal mean-field saddle map.

    Built like ``_build_fluct_map``.
    """
    out: dict[str, str] = {}
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


def _field_units(model: dict, name: str):
    """``(units, population)`` of the physical field ``name`` (its plain
    name ``x``, its fluctuation name ``dx`` or its response ``xt``), or
    ``None`` when the model has no such field."""
    pfs = model.get('physical_fields') or []
    pf = next((f for f in pfs if name in (f.get('natural_name'), f.get('name'))), None)
    if pf is None and name.endswith('t'):
        pf = next((f for f in pfs if name[:-1] in (f.get('natural_name'), f.get('name'))), None)
    if pf is None:
        return None
    pop = pf.get('population')
    if pop:
        for p in (model.get('populations') or []):
            if p.get('name') == pop:
                return max(1, int(p.get('size') or 1)), pop
    if pf.get('indexed'):
        idx = (model.get('index_sets') or {}).get('pop')
        if idx:
            return max(1, len(idx)), 'pop'
    return 1, pop


def resolve_external_fields(model: dict, external_fields) -> list:
    """External fields with bare names resolved: a field name alone
    (``'x'``, or ``'dx'``) is the leg ``('x', 1)`` of a field with one unit,
    so ``['x', 'x']`` is ``[('x', 1), ('x', 1)]`` (the same run, byte for
    byte).  A field whose population has several units needs the unit,
    ``('x', i)``; a bare name for it raises ``ValueError``.  Pairs pass
    through unchanged; a name the model does not have is left to the
    later checks.  Without a bare name the input is returned as it is."""
    if external_fields is None or not any(isinstance(e, str) for e in external_fields):
        return external_fields
    out = []
    for e in external_fields:
        if not isinstance(e, str):
            out.append(e)
            continue
        info = _field_units(model, e)
        if info is not None and info[0] > 1:
            raise ValueError(
                f'external field {e!r}: the field lives in population {info[1]!r} of '
                f'{info[0]} units, so a bare name does not say which unit; give '
                f'({e!r}, i) with 1 <= i <= {info[0]}, e.g. [({e!r}, 1), ({e!r}, 2)].')
        out.append((e, 1))
    return out


def normalize_external_fields(external_fields,
                              naming_convention: Optional[dict] = None):
    """Translate user-facing external fields to internal fluctuation
    names::

        normalize_external_fields([('n', 1)])              → [('dn', 1)]
        normalize_external_fields([('r', 1)],              # custom letter
            naming_convention={'fluctuation_fields': {'r': 'dr'}})
                                                            → [('dr', 1)]

    Already-internal names pass through unchanged (and are the only
    names understood when ``naming_convention`` is omitted).
    """
    fluct_map = _build_fluct_map(naming_convention)
    out = []
    for entry in external_fields:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            raise ValueError(
                "external_fields entries must be (field, population) pairs; "
                f"got {entry!r}.  E.g. [('x', 1), ('x', 1)].")
        name, pop = entry
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
            if not 1 <= pop <= len(vals):
                raise IndexError(
                    f'population index {pop} out of range for {name!r} '
                    f'(valid 1..{len(vals)}); indices are 1-based.')
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
                ii = int(i)
                if hasattr(val, '__len__') and not 1 <= ii <= len(val):
                    raise IndexError(
                        f'index {ii} out of range for parameter {name!r} '
                        f'(valid 1..{len(val)}); indices are 1-based.')
                val = val[ii - 1]      # 1-based
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

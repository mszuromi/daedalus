"""
One naming service for mean-field saddles and physical fields.

A parameter is a mean-field saddle iff the model declares it with
``mean_field=True`` (the builder's flag; ``physical_field(...)`` creates
``<natural>star`` with the flag set, and a model may declare its saddles
under any other name).  The field a saddle belongs to is the saddle's
``natural_name``.  Nothing here reads a spelling: a real parameter named
``tstar`` is an ordinary parameter, and a saddle named ``rate_mf`` is a
saddle.  (Audit GEN-03/GEN-04: the suffix ``star`` and the letters
``n``/``v``/``m`` used to decide this in several places.)
"""
from __future__ import annotations

import re
from typing import Optional


def _params(model: Optional[dict]) -> list:
    return list((model or {}).get('parameters') or [])


def is_saddle(pspec: dict) -> bool:
    """Whether the parameter spec declares a mean-field saddle."""
    return bool(pspec.get('mean_field'))


def saddle_names(model: Optional[dict]) -> list[str]:
    """Names of the model's mean-field saddle parameters (declaration
    order)."""
    return [str(p['name']) for p in _params(model) if is_saddle(p)]


def free_parameter_names(model: Optional[dict]) -> list[str]:
    """Names of the declared parameters that are not saddles (the ones a
    run supplies or overrides)."""
    return [str(p['name']) for p in _params(model) if not is_saddle(p)]


def natural_name(field: dict) -> str:
    """User-facing name of a physical-field dict (its ``natural_name``,
    else its internal name)."""
    return field.get('natural_name') or field['name']


def _field(model: dict, name: str) -> Optional[dict]:
    for f in (model.get('physical_fields') or []):
        if name in (f.get('natural_name'), f.get('name')):
            return f
    return None


def field_natural_name(model: dict, field: str) -> str:
    """Natural name of the physical field given by its natural or internal
    name (``field`` itself when the model has no such field)."""
    f = _field(model, field)
    return natural_name(f) if f is not None else field


def saddle_for_field(model: dict, field: str) -> Optional[str]:
    """The saddle parameter of a physical field given by its natural or
    internal name: the saddle whose ``natural_name`` is the field's natural
    name.  ``None`` when the model declares none."""
    f = _field(model, field)
    nat = natural_name(f) if f is not None else field
    for p in _params(model):
        if is_saddle(p) and p.get('natural_name') == nat:
            return str(p['name'])
    return None


def field_for_saddle(model: dict, saddle: str) -> Optional[str]:
    """Natural name of the physical field a saddle parameter belongs to
    (its ``natural_name``), ``None`` when it names none."""
    for p in _params(model):
        if is_saddle(p) and p['name'] == saddle:
            nat = p.get('natural_name')
            return str(nat) if nat else None
    return None


def saddle_domain(model: dict, saddle: str) -> Optional[str]:
    """Declared ``domain`` of a saddle parameter."""
    for p in _params(model):
        if p['name'] == saddle:
            return p.get('domain')
    return None


def is_saddle_symbol(name: str, model: Optional[dict]) -> bool:
    """Whether the symbol ``name`` is a saddle value: a declared saddle,
    plain or with its population index appended (``xstar`` / ``xstar1`` /
    ``nEstar2``)."""
    return any(re.fullmatch(re.escape(b) + r'\d*', name)
               for b in saddle_names(model))


def is_parameter_symbol(name: str, model: Optional[dict]) -> bool:
    """Whether the symbol ``name`` is the value of a declared free (non-
    saddle) parameter, plain or with its indices appended (``w`` /
    ``w12``).  Derived symbols a solve also reports (a Taylor coefficient
    ``phi1_1``, a formal function's saddle value ``phi0_1``) are neither
    parameters nor saddles (audit GEN-17)."""
    return any(re.fullmatch(re.escape(b) + r'\d*', name)
               for b in free_parameter_names(model))


def saddle_label(model: dict, saddle: str) -> str:
    """Short display label of a saddle: ``<field>*`` for a field's saddle,
    else the parameter name."""
    nat = field_for_saddle(model, saddle)
    return f'{nat}*' if nat else saddle

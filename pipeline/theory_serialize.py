"""
pipeline.theory_serialize — emit a ``.theory.py`` file from a form-
state spec dict.

The theory-input UI collects user input as a flat dict; this module
turns that dict into the on-disk format consumed by
``notebooks/theory_runner.ipynb`` (and any other loader that imports
the file as a Python module).

Output file shape::

    \"\"\"<docstring>\"\"\"
    from pipeline.theory import TheoryBuilder


    def build():
        return (
            TheoryBuilder(<name>, n_populations=<n>)
            .response_field(...)
            .physical_field(..., natural_name=...)
            .parameter(..., mean_field=True, natural_name=...)
            .define_function(...)
            .define_kernel(...)
            .declare_cgf_term(...)
            .set_action_text(\"\"\"...\"\"\")
            .set_mf_equation(..., \"...\")
            .build()
        )


    DEFAULT_FUNDAMENTAL = {...}
    METADATA = {...}

The same module exposes :func:`load_spec_from_file` which reverses
the operation by importing the file and reconstructing the spec dict
— useful for the UI's "Load existing theory" feature.
"""
from __future__ import annotations

from typing import Any
import re


# ── Python-source helpers ─────────────────────────────────────────────

def _py_repr(value: Any) -> str:
    """``repr`` with sensible formatting for primitive theory data.

    Uses Python defaults for scalars / strings, but pretty-prints
    nested lists across lines for readability.
    """
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        # 2D list (matrix) → multi-line
        if value and all(isinstance(row, list) for row in value):
            inner = ',\n'.join('    ' + _py_repr(row) for row in value)
            return f'[\n{inner},\n]'
        # 1D list → single line if short, otherwise wrapped
        body = ', '.join(_py_repr(v) for v in value)
        if len(body) <= 60:
            return f'[{body}]'
        items = ',\n'.join(f'    {_py_repr(v)}' for v in value)
        return f'[\n{items},\n]'
    if isinstance(value, dict):
        if not value:
            return '{}'
        items = ',\n'.join(
            f'    {_py_repr(k)}: {_py_repr(v)}'
            for k, v in value.items())
        return '{\n' + items + ',\n}'
    if isinstance(value, tuple):
        body = ', '.join(_py_repr(v) for v in value)
        return f'({body})' + (',' if len(value) == 1 else '')
    return repr(value)


def _kw(name: str, value: Any) -> str:
    """Format a single ``name=value`` keyword pair."""
    return f'{name}={_py_repr(value)}'


def _kw_chain(*pairs: tuple[str, Any]) -> str:
    """Format keyword pairs into a comma-joined string, omitting
    pairs whose value is ``None`` or an empty string."""
    parts = []
    for name, value in pairs:
        if value is None or value == '':
            continue
        parts.append(_kw(name, value))
    return ', '.join(parts)


def _slugify(name: str) -> str:
    """Filesystem-safe lowercase identifier."""
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower() or 'theory'


# ── Builder-method emitters ───────────────────────────────────────────

def _emit_response_field(f: dict) -> str:
    return '.response_field(' + _kw_chain(
        ('', None),  # placeholder; first arg is positional
    ) + (
        f'{_py_repr(f["name"])}, '
        f'{_kw_chain(("indexed", f.get("indexed", True)), ("latex", f.get("latex")), ("description", f.get("description")))})'
    )


def _emit_field(method: str, f: dict, *, with_natural: bool) -> str:
    """Emit a ``.<method>('name', kwarg=...)`` line.

    For physical fields (``method='physical_field'``), the ``name``
    arg is treated as the user-facing natural letter — TheoryBuilder
    auto-prefixes ``d`` to get the internal fluctuation name and
    auto-creates the response field + saddle parameter.
    """
    pairs = [('indexed', f.get('indexed', True))]
    if with_natural:
        # Only include natural_name if it differs from name (legacy
        # support for files that explicitly declare the d-prefixed
        # internal name); the new style omits natural_name entirely.
        nn = f.get('natural_name')
        if nn and nn != f['name']:
            pairs.append(('natural_name', nn))
    pairs += [
        ('latex',       f.get('latex')),
        ('description', f.get('description')),
    ]
    kwargs = _kw_chain(*pairs)
    head = f'.{method}({_py_repr(f["name"])}'
    return head + (', ' + kwargs if kwargs else '') + ')'


def _emit_parameter(p: dict) -> str:
    """Emit a ``.parameter(...)`` call.

    Saddle parameters (``mean_field=True``) are NOT emitted — the
    framework auto-creates them when their physical field is
    declared.  We filter those out at the spec collection layer.
    """
    # Map UI 'type' to TheoryBuilder's indexed= argument
    ptype = p.get('type', 'scalar')
    if ptype == 'scalar':
        indexed_kw = None     # default False
    elif ptype in ('vector',):
        indexed_kw = True
    elif ptype == 'matrix':
        indexed_kw = 'matrix'
    else:
        indexed_kw = p.get('indexed')   # passthrough

    kwargs = _kw_chain(
        ('default',      p.get('default')),
        ('indexed',      indexed_kw),
        ('domain',       p.get('domain')),
        ('mean_field',   p.get('mean_field') or None),
        ('natural_name', p.get('natural_name')),
        ('description',  p.get('description')),
    )
    head = f'.parameter({_py_repr(p["name"])}'
    return head + (', ' + kwargs if kwargs else '') + ')'


def _emit_function(fn: dict) -> str:
    kwargs = _kw_chain(
        ('args',        list(fn['args'])),
        ('expression',  fn['expression']),
        ('latex',       fn.get('latex')),
        ('description', fn.get('description')),
    )
    return f'.define_function({_py_repr(fn["name"])}, {kwargs})'


def _emit_kernel(k: dict) -> str:
    kwargs = _kw_chain(
        ('time_expr',   k.get('time_expr')),
        ('freq_image',  k.get('freq_image')),
        ('latex_name',  k.get('latex_name') or k.get('latex')),
        ('sage_name',   k.get('sage_name')),
        ('description', k.get('description')),
    )
    head = f'.define_kernel({_py_repr(k["name"])}'
    return head + (', ' + kwargs if kwargs else '') + ')'


def _emit_cgf_term(c: dict) -> str:
    kwargs = _kw_chain(
        ('order',          int(c['order'])),
        ('coefficient',    c['coefficient']),
        ('kernel',         c.get('kernel')),
    )
    return (f'.declare_cgf_term({_py_repr(c["name"])}, '
            f'{_py_repr(c["response_field"])}, {kwargs})')


def _emit_action(action_text: str) -> str:
    """Emit ``.set_action_text('''...''')`` with sensible indentation."""
    body = (action_text or '').strip()
    if '\n' in body:
        # Multi-line — preserve as triple-quoted block, indented one level
        indented = '\n'.join('    ' + line for line in body.splitlines())
        return f".set_action_text('''\n{indented}\n''')"
    return f'.set_action_text({_py_repr(body)})'


def _emit_mf_equation(saddle: str, rhs: str) -> str:
    return (f'.set_mf_equation({_py_repr(saddle)}, '
            f'{_py_repr(rhs)})')


# ── Main entry points ─────────────────────────────────────────────────

def render_theory_file(spec: dict) -> str:
    """Render the full ``.theory.py`` source from a spec dict."""
    name = spec['name']
    n_pop = int(spec.get('n_populations', 1))
    description = spec.get('description', '')
    response_fields = spec.get('response_fields', []) or []
    physical_fields = spec.get('physical_fields', []) or []
    parameters      = spec.get('parameters', [])      or []
    # Auto-filter saddle parameters and parameters whose names match
    # the auto-generated saddle naming convention (<natural>star) —
    # the framework re-creates these from the physical-field
    # declarations.
    physical_natural = {f.get('natural_name') or f['name']
                        for f in physical_fields}
    auto_saddle_names = {f'{nat}star' for nat in physical_natural}
    parameters = [p for p in parameters
                  if not p.get('mean_field')
                  and p['name'] not in auto_saddle_names]
    functions       = spec.get('functions', [])       or []
    kernels         = spec.get('kernels', [])         or []
    cgf_terms       = spec.get('cgf_terms', [])       or []
    action_text     = spec.get('action_text', '')
    mf_equations    = spec.get('mf_equations', [])    or []
    default_fund    = spec.get('default_fundamental', {}) or {}
    metadata        = spec.get('metadata', {})        or {}

    # Header
    docstring_lines = [f'"""', f'{name} — text-driven theory file.']
    if description:
        docstring_lines += ['', description]
    docstring_lines += [
        '',
        'Generated by the theory-input UI '
        '(``notebooks/theory_builder.ipynb``).',
        'Loaded by ``notebooks/theory_runner.ipynb``.',
        '"""',
    ]

    out = []
    out.append('\n'.join(docstring_lines))
    out.append('from pipeline.theory import TheoryBuilder')
    out.append('')
    out.append('')
    out.append('def build():')
    out.append(f'    return (')
    out.append(f'        TheoryBuilder({_py_repr(name)}, n_populations={n_pop})')

    # Response fields: only emit if user explicitly declared them.
    # The new natural-name style relies on TheoryBuilder.physical_field
    # to auto-create the conjugate response.
    for f in response_fields:
        out.append(f'        {_emit_field("response_field", f, with_natural=False)}')
    for f in physical_fields:
        out.append(f'        {_emit_field("physical_field", f, with_natural=True)}')
    for p in parameters:
        out.append(f'        {_emit_parameter(p)}')
    for fn in functions:
        out.append(f'        {_emit_function(fn)}')
    for k in kernels:
        out.append(f'        {_emit_kernel(k)}')
    for c in cgf_terms:
        out.append(f'        {_emit_cgf_term(c)}')

    if action_text:
        # Indent the action method itself one level (under the chain)
        action_call = _emit_action(action_text)
        # Add indentation for each line of the action call
        for i, line in enumerate(action_call.splitlines()):
            prefix = '        ' if i == 0 else '        '
            out.append(prefix + line)

    for eq in mf_equations:
        out.append(f'        {_emit_mf_equation(eq["saddle"], eq["rhs"])}')

    out.append('        .build()')
    out.append('    )')
    out.append('')
    out.append('')
    out.append(f'DEFAULT_FUNDAMENTAL = {_py_repr(default_fund)}')
    out.append('')
    out.append('')
    out.append(f'METADATA = {_py_repr(metadata)}')
    out.append('')

    return '\n'.join(out)


def save_theory_to_file(spec: dict, path: str) -> str:
    """Write the spec to ``path`` (overwriting if it exists).

    Returns the path written.  If ``path`` is a directory, the
    filename is derived from ``spec['name']`` via :func:`_slugify`
    plus ``.theory.py``.
    """
    import os

    if os.path.isdir(path):
        path = os.path.join(path, f'{_slugify(spec["name"])}.theory.py')
    elif not path.endswith('.theory.py'):
        # User passed a base name — append extension
        if not path.endswith('.py'):
            path = path + '.theory.py'

    source = render_theory_file(spec)
    with open(path, 'w') as f:
        f.write(source)
    return path


# ── Reverse direction (file → spec dict) ──────────────────────────────

def load_spec_from_file(path: str) -> dict:
    """Reconstruct a spec dict by importing ``path`` and walking the
    TheoryBuilder it constructs.

    Useful for the UI's "Load existing theory" button — re-populates
    the form from a saved file.

    Caveat: this round-trips the SHAPE (fields, parameters, action
    text, etc.) but NOT cosmetic details that don't reach the
    builder's internal state (e.g. comments in the source file).
    """
    import importlib.util, os

    name = os.path.basename(path)
    if name.endswith('.theory.py'):
        name = name[:-len('.theory.py')]
    elif name.endswith('.py'):
        name = name[:-3]

    spec_loader = importlib.util.spec_from_file_location(
        f'_loaded_theory.{name}', path)
    mod = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(mod)

    # Build the model just to populate the builder's internal state.
    # We can't get the builder back from .build() (it returns a dict),
    # so we monkey-patch — call build and capture the model dict, then
    # walk the model dict to recover the spec.  Lossy on a few extras.
    model = mod.build()

    # Recover physical_fields with natural names
    pf = []
    for f in model.get('physical_fields', []):
        entry = {'name': f['name']}
        for k in ('indexed', 'natural_name', 'latex', 'description'):
            if k in f:
                entry[k] = f[k]
        pf.append(entry)

    rf = [{'name': f['name'],
           **{k: f[k] for k in ('indexed', 'latex', 'description') if k in f}}
          for f in model.get('response_fields', [])]

    # Parameters
    params = []
    for p in model.get('parameters', []):
        ptype = ('matrix' if p.get('indexed') == 'matrix'
                 else 'vector' if p.get('indexed') is True
                 else 'scalar')
        entry = {'name': p['name'], 'type': ptype}
        for k in ('domain', 'mean_field', 'natural_name', 'description'):
            if k in p:
                entry[k] = p[k]
        params.append(entry)

    # Kernels
    kernels = [
        {'name': k['name'],
         **{kk: k[kk] for kk in ('latex_name', 'sage_name', 'description')
            if kk in k}}
        for k in model.get('kernels', [])
    ]

    # The text strings (action / mf eqs / fn exprs / cgf coeffs) live in
    # the closures attached to the model lambdas; we have to read them
    # from the (still-importable) module's source if possible.  For now,
    # don't try — the UI's "Load" feature can use a sidecar JSON file
    # if perfect round-tripping is needed.

    return {
        'name':            model.get('name', ''),
        'n_populations':   len(model.get('index_sets', {}).get('pop', [])),
        'description':     getattr(mod, '__doc__', '') or '',
        'response_fields': rf,
        'physical_fields': pf,
        'parameters':      params,
        'functions':       [],   # see caveat above
        'kernels':         kernels,
        'cgf_terms':       [],
        'action_text':     '',
        'mf_equations':    [],
        'default_fundamental': getattr(mod, 'DEFAULT_FUNDAMENTAL', {}),
        'metadata':        getattr(mod, 'METADATA', {}),
    }

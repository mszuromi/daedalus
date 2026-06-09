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
            .equation(lhs=\"...\", rhs=\"...\", population=\"E\")
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

    For heterogeneous-population theories, ``f['population']`` names
    the population this field belongs to; the emitter passes it
    through as ``population='<name>'``.
    """
    pairs = []
    # Population annotation (new style).  Mutually exclusive with
    # ``indexed`` — if a population is given, the field's shape is
    # determined by size(population) and ``indexed=True`` is implied.
    if f.get('population'):
        pairs.append(('population', f['population']))
    else:
        pairs.append(('indexed', f.get('indexed', True)))
    if with_natural:
        nn = f.get('natural_name')
        if nn and nn != f['name']:
            pairs.append(('natural_name', nn))
    pairs += [
        ('latex',       f.get('latex')),
        ('description', f.get('description')),
    ]
    # Spatial dimension (per-field, v1).  Emitted only when non-zero
    # so time-only fields round-trip unchanged.  Round-tripping
    # per-field preserves mixed dim=0/dim=d theories without needing
    # to also emit the ``.spatial_dim(d)`` convenience call.
    if f.get('spatial_dim'):
        pairs.append(('spatial_dim', int(f['spatial_dim'])))
    kwargs = _kw_chain(*pairs)
    head = f'.{method}({_py_repr(f["name"])}'
    return head + (', ' + kwargs if kwargs else '') + ')'


def _emit_parameter(p: dict) -> str:
    """Emit a ``.parameter(...)`` call.

    Saddle parameters (``mean_field=True``) are NOT emitted — the
    framework auto-creates them when their physical field is
    declared.

    For heterogeneous-population theories, the spec carries an
    ``indexed_by`` list (zero / one / two population names).
    Legacy ``type`` ('scalar' / 'vector' / 'matrix') still works via
    a fallback translation.
    """
    indexed_by = p.get('indexed_by')
    if indexed_by:                       # new-style annotation wins
        indexed_kw = None                # don't emit ``indexed=``
    else:                                # legacy translation
        ptype = p.get('type', 'scalar')
        if ptype in ('scalar',):
            indexed_kw = None
        elif ptype in ('vector',):
            indexed_kw = True
        elif ptype == 'matrix':
            indexed_kw = 'matrix'
        else:
            indexed_kw = p.get('indexed')
    kwargs = _kw_chain(
        ('default',      p.get('default')),
        ('indexed_by',   list(indexed_by) if indexed_by else None),
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
        ('population',  fn.get('population')),
        ('latex',       fn.get('latex')),
        ('description', fn.get('description')),
    )
    return f'.define_function({_py_repr(fn["name"])}, {kwargs})'


def _emit_kernel(k: dict) -> str:
    """Emit a ``.define_kernel(...)`` call.

    Heterogeneous-population spec uses ``indexed_by=['pop1', 'pop2']``;
    legacy spec uses ``indexed='vector'`` / ``'matrix'``.  Either is
    accepted — ``indexed_by`` wins if both are present.
    """
    indexed_by = k.get('indexed_by')
    if indexed_by:
        indexed_value = None
    else:
        indexed = k.get('indexed')
        if indexed in (None, False, 'scalar'):
            indexed_value = None
        elif indexed is True or indexed == 'vector':
            indexed_value = 'vector'
        elif indexed == 'matrix':
            indexed_value = 'matrix'
        else:
            indexed_value = indexed
    kwargs = _kw_chain(
        ('time_expr',   k.get('time_expr')),
        ('freq_image',  k.get('freq_image')),
        ('latex_name',  k.get('latex_name') or k.get('latex')),
        ('sage_name',   k.get('sage_name')),
        ('description', k.get('description')),
        ('indexed_by',  list(indexed_by) if indexed_by else None),
        ('indexed',     indexed_value),
    )
    head = f'.define_kernel({_py_repr(k["name"])}'
    return head + (', ' + kwargs if kwargs else '') + ')'


def _emit_cgf_term(c: dict) -> str:
    """Render one CGF row as a ``.declare_cgf_term(...)`` call.

    Two shapes:
      * Legacy single-leg: ``response_field='mt'`` positional+kw.
      * New multi-leg: ``response_legs=['xt', 'yt']`` as a keyword.
        Single-leg specs that round-trip through the new
        ``response_legs`` key (e.g. ``['mt']``) also render via the
        keyword form for consistency.

    Selection: if ``response_legs`` is set AND has > 1 entry, render
    via the keyword form.  Otherwise emit the legacy positional
    ``response_field`` for back-compat with older callers.

    Markovianize override: when the row stores
    ``markovianize=True`` or ``markovianize=False`` (i.e. an explicit
    opt-in / opt-out for the colored-noise preprocessor), emit the
    keyword.  Rows with the default ``None`` / ``'auto'`` are
    serialized without the keyword so existing files stay clean.
    """
    legs = c.get('response_legs')
    markov_extra: list[tuple] = []
    mk = c.get('markovianize')
    if isinstance(mk, bool):
        markov_extra.append(('markovianize', mk))
    base_kwargs = _kw_chain(
        ('order',          int(c['order'])),
        ('coefficient',    c['coefficient']),
        ('kernel',         c.get('kernel')),
        *markov_extra,
    )
    if legs and isinstance(legs, (list, tuple)) and len(legs) > 1:
        return (f'.declare_cgf_term({_py_repr(c["name"])}, '
                f'response_legs={_py_repr(list(legs))}, {base_kwargs})')
    # Legacy single-leg shape.  ``response_field`` is the first leg if
    # legs is a 1-element list, else use the explicit field.
    if legs and isinstance(legs, (list, tuple)) and len(legs) == 1:
        single = legs[0]
    else:
        single = c.get('response_field', '')
    return (f'.declare_cgf_term({_py_repr(c["name"])}, '
            f'{_py_repr(single)}, {base_kwargs})')


def _emit_action(action_text: str) -> str:
    """Emit ``.set_action_text('''...''')`` with sensible indentation.

    Uses :func:`textwrap.dedent` before re-indenting so a load → save
    cycle is whitespace-stable (without dedent, every save would
    accumulate an extra four spaces on each subsequent line).
    """
    import textwrap
    body = textwrap.dedent(action_text or '').strip()
    if '\n' in body:
        indented = '\n'.join('    ' + line for line in body.splitlines())
        return f".set_action_text('''\n{indented}\n''')"
    return f'.set_action_text({_py_repr(body)})'


def _emit_mf_equation(saddle: str, rhs: str) -> str:
    return (f'.set_mf_equation({_py_repr(saddle)}, '
            f'{_py_repr(rhs)})')


def _emit_equation(lhs: str, rhs: str, population) -> str:
    """Emit a ``.equation(lhs=..., rhs=..., population=...)`` call for
    the DAE-style MF spec.  Used by the new (Tab 9) UI MF form."""
    lhs_repr = _py_repr(lhs)
    rhs_repr = _py_repr(rhs)
    parts = [f'lhs={lhs_repr}', f'rhs={rhs_repr}']
    if population:
        parts.append(f'population={_py_repr(population)}')
    return f'.equation({", ".join(parts)})'


# ── Main entry points ─────────────────────────────────────────────────

def render_theory_file(spec: dict) -> str:
    """Render the full ``.theory.py`` source from a spec dict.

    Heterogeneous-population specs carry a ``populations`` list
    (``[{'name': 'E', 'size': N_E}, ...]``).  In that case the
    emitted file makes ``TheoryBuilder()`` without ``n_populations=``
    and then chains one ``.population('name', size=N)`` call per
    declared population.  Legacy specs (only ``n_populations`` set)
    keep the old ``TheoryBuilder('name', n_populations=N)`` form.
    """
    name = spec['name']
    populations = spec.get('populations') or []
    # If populations is empty, fall back to the legacy n_populations
    # path so old saved specs keep loading.
    n_pop = (len(populations) if populations
             else int(spec.get('n_populations', 1)))
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
    # New DAE-form list: {lhs, rhs, population} records.  When
    # populated, the file emits .equation(...) calls; when empty,
    # falls back to the legacy .set_mf_equation(...) emission from
    # ``mf_equations`` for backward compat.
    equations       = spec.get('equations', [])       or []
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

    # Builder-split Phase 1: emit the domain-specific forward builder so a
    # round-tripped file declares its kind explicitly.  Spatial iff any physical
    # field carries spatial_dim>=1; otherwise temporal.  The back-compat
    # TheoryBuilder shim still loads either (load_spec_from_file accepts all three
    # constructor names).
    _is_spatial = any(int(f.get('spatial_dim', 0) or 0) > 0
                      for f in physical_fields)
    builder_cls = 'SpatialTheoryBuilder' if _is_spatial else 'TemporalTheoryBuilder'

    out = []
    out.append('\n'.join(docstring_lines))
    out.append(f'from pipeline.theory import {builder_cls}')
    out.append('')
    out.append('')
    out.append('def build():')
    out.append(f'    return (')
    if populations:
        # Heterogeneous populations: just name in the constructor,
        # populations come via chained .population(...) calls so each
        # one's size is explicit and the declaration order is visible.
        out.append(f'        {builder_cls}({_py_repr(name)})')
        for pop in populations:
            pop_kwargs = _kw_chain(
                ('size',        int(pop.get('size', 1))),
                ('description', pop.get('description')),
            )
            head = f'.population({_py_repr(pop["name"])}'
            line = head + (', ' + pop_kwargs if pop_kwargs else '') + ')'
            out.append(f'        {line}')
    else:
        out.append(
            f'        {builder_cls}({_py_repr(name)}, n_populations={n_pop})')

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

    # Prefer the new DAE-form .equation(...) emission when available.
    # Otherwise fall back to legacy .set_mf_equation(...) for old specs.
    if equations:
        for eq in equations:
            out.append(
                f'        {_emit_equation(eq["lhs"], eq["rhs"], eq.get("population"))}'
            )
    else:
        for eq in mf_equations:
            out.append(
                f'        {_emit_mf_equation(eq["saddle"], eq["rhs"])}'
            )

    # Stability-analysis toggle (default OFF on the TheoryBuilder).
    # Only emit when explicitly ON so OFF-by-default theories stay
    # textually quiet.  Allows a theory file to be diffed cleanly
    # against the UI's intent.
    if spec.get('stability_analysis'):
        out.append('        .stability_analysis(True)')

    # Markovianize default (default ON on the TheoryBuilder).  Only
    # emit when explicitly OFF so theories that accept the default
    # behaviour stay textually quiet.  Round-trips
    # ``.markovianize(False)`` for users who opted out at the builder
    # level (e.g. their colored kernel doesn't match the v1
    # single-Lorentzian template).
    if spec.get('markovianize_default') is False:
        out.append('        .markovianize(False)')

    # Spatial boundary / initial conditions (v1).  ``spatial_dim`` is
    # carried per-field (round-trips via _emit_field above), so only
    # the BC/IC declarations need their own emit lines.  Emitted only
    # when present — time-only theories carry neither key and stay
    # textually quiet.
    bc = spec.get('boundary')
    if bc and bc.get('mode'):
        bc_pairs = [(k, v) for k, v in bc.items() if k != 'mode']
        bc_kwargs = _kw_chain(*bc_pairs)
        line = f'        .boundary({_py_repr(bc["mode"])}'
        out.append(line + (', ' + bc_kwargs if bc_kwargs else '') + ')')
    ic = spec.get('initial')
    if ic and ic.get('mode'):
        ic_pairs = [(k, v) for k, v in ic.items() if k != 'mode']
        ic_kwargs = _kw_chain(*ic_pairs)
        line = f'        .initial({_py_repr(ic["mode"])}'
        out.append(line + (', ' + ic_kwargs if ic_kwargs else '') + ')')

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
    """Parse a ``.theory.py`` file's source AST and reconstruct the
    spec dict it was generated from.

    Reads the source directly rather than executing it, so:
      * the action text, mf-equation RHS strings, function
        expressions, kernel time/freq exprs all round-trip
        verbatim — they're stored as string literals in the source;
      * Python literals in default-values, indexed_by lists, etc.
        round-trip exactly via ``ast.literal_eval``;
      * comments and formatting are lost (this is an AST round-trip,
        not a textual one);
      * a malformed file raises ``SyntaxError`` from ``ast.parse``.

    Used by ``TheoryUI.load(path)`` to re-populate the form so the
    user can edit a previously-saved theory without rewriting it.
    """
    import ast
    import os

    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)

    # ── Module docstring → description ────────────────────────────
    description = ast.get_docstring(tree) or ''
    # Strip the boilerplate header that render_theory_file inserts so
    # only the user-supplied description remains.
    description = _strip_doc_boilerplate(description)

    # ── Module-level DEFAULT_FUNDAMENTAL / METADATA ───────────────
    default_fund: dict = {}
    metadata: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name):
                continue
            try:
                val = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                val = {}
            if tgt.id == 'DEFAULT_FUNDAMENTAL':
                default_fund = val if isinstance(val, dict) else {}
            elif tgt.id == 'METADATA':
                metadata = val if isinstance(val, dict) else {}

    # ── Find the build() function and its return-expression chain ─
    build_func = next((n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == 'build'),
                      None)
    if build_func is None:
        raise ValueError(f'{path}: no build() function found.')
    ret = next((s for s in build_func.body if isinstance(s, ast.Return)),
               None)
    if ret is None or not isinstance(ret.value, ast.Call):
        raise ValueError(f'{path}: build() does not return a call expression.')

    # Walk the chain of attribute-calls back to the constructor.
    chain = []
    cur = ret.value
    while isinstance(cur, ast.Call):
        chain.append(cur)
        if isinstance(cur.func, ast.Attribute):
            cur = cur.func.value
        elif isinstance(cur.func, ast.Name):
            break
        else:
            break
    chain.reverse()    # constructor first, then method calls in order

    # ── Initialise the spec dict ──────────────────────────────────
    spec: dict = {
        'name':            '',
        'description':     description,
        'populations':     [],
        'n_populations':   0,
        'response_fields': [],
        'physical_fields': [],
        'parameters':      [],
        'functions':       [],
        'kernels':         [],
        'cgf_terms':       [],
        'action_text':     '',
        'mf_equations':    [],
        # New DAE form: list of {lhs, rhs, population} records.
        # Populated below when the source uses ``.equation(...)``
        # calls.  Old files using ``.set_mf_equation(...)`` populate
        # ``mf_equations`` instead and the UI auto-converts on load.
        'equations':       [],
        # Stability-analysis toggle from ``.stability_analysis(bool)``;
        # default OFF.  Stays False until the parser hits an explicit
        # ``.stability_analysis(True)`` call.
        'stability_analysis': False,
        # Markovianize-preprocessor builder-level toggle from
        # ``.markovianize(bool)``; default ON.  Stays True until the
        # parser hits an explicit ``.markovianize(False)`` call.  See
        # ``pipeline/colored_to_markovian.py`` and
        # ``docs/correlated_noise_capabilities.md`` §1.5.
        'markovianize_default': True,
        'default_fundamental': default_fund,
        'metadata':        metadata,
    }

    def _lit(node, default=None):
        """``ast.literal_eval`` with a fallback."""
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return default

    def _kwargs(call):
        return {kw.arg: _lit(kw.value) for kw in call.keywords if kw.arg}

    for call in chain:
        # Constructor: TheoryBuilder('name', n_populations=N)
        if isinstance(call.func, ast.Name) and call.func.id in (
                'TheoryBuilder', 'SpatialTheoryBuilder', 'TemporalTheoryBuilder'):
            if call.args:
                spec['name'] = _lit(call.args[0], '')
            kw = _kwargs(call)
            if 'n_populations' in kw:
                spec['n_populations'] = int(kw['n_populations'])
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        method = call.func.attr
        kw = _kwargs(call)
        args = [_lit(a) for a in call.args]

        if method == 'population':
            entry = {'name': args[0] if args else ''}
            for k in ('size', 'description'):
                if k in kw:
                    entry[k] = kw[k]
            spec['populations'].append(entry)

        elif method == 'physical_field':
            entry = {'name': args[0] if args else ''}
            for k in ('indexed', 'population', 'natural_name',
                      'latex', 'description', 'spatial_dim'):
                if k in kw:
                    entry[k] = kw[k]
            # Inherit a builder-level default set by an EARLIER
            # ``.spatial_dim(d)`` call (the convenience method may
            # precede field declarations in hand-written files).
            # Explicit ``spatial_dim=`` kwargs above still win.
            if 'spatial_dim' not in entry and spec.get('_default_spatial_dim'):
                entry['spatial_dim'] = spec['_default_spatial_dim']
            spec['physical_fields'].append(entry)

        elif method == 'response_field':
            entry = {'name': args[0] if args else ''}
            for k in ('indexed', 'latex', 'description'):
                if k in kw:
                    entry[k] = kw[k]
            spec['response_fields'].append(entry)

        elif method == 'parameter':
            entry = {'name': args[0] if args else ''}
            for k in ('default', 'indexed_by', 'indexed', 'domain',
                      'mean_field', 'natural_name', 'description'):
                if k in kw:
                    entry[k] = kw[k]
            spec['parameters'].append(entry)

        elif method == 'define_function':
            entry = {'name': args[0] if args else ''}
            for k in ('args', 'expression', 'population',
                      'latex', 'description'):
                if k in kw:
                    entry[k] = kw[k]
            spec['functions'].append(entry)

        elif method == 'define_kernel':
            entry = {'name': args[0] if args else ''}
            for k in ('time_expr', 'freq_image', 'latex_name',
                      'sage_name', 'description', 'indexed_by', 'indexed'):
                if k in kw:
                    entry[k] = kw[k]
            spec['kernels'].append(entry)

        elif method == 'declare_cgf_term':
            entry: dict = {}
            # Two emission shapes round-trip cleanly:
            #   Legacy:   .declare_cgf_term('X', 'mt', order=..., ...)
            #   Multi-leg: .declare_cgf_term('X', response_legs=['xt','yt'],
            #                                order=..., ...)
            # Either name is positional (args[0]); response_field is
            # positional in the legacy form OR keyword in the
            # multi-leg form (or absent entirely when response_legs is
            # present).
            if len(args) >= 1:
                entry['name'] = args[0]
            if 'name' in kw:
                entry['name'] = kw['name']
            if len(args) >= 2:
                entry['response_field'] = args[1]
            if 'response_field' in kw:
                entry['response_field'] = kw['response_field']
            if 'response_legs' in kw:
                legs = kw['response_legs']
                # Accept either a list literal or a string with commas.
                if isinstance(legs, str):
                    legs = [s.strip() for s in legs.split(',') if s.strip()]
                entry['response_legs'] = list(legs) if legs else None
            for k in ('order', 'coefficient', 'kernel'):
                if k in kw:
                    entry[k] = kw[k]
            # Markovianize override (per-row): True / False explicit
            # values round-trip; absence stays 'auto' (= None).
            if 'markovianize' in kw:
                entry['markovianize'] = kw['markovianize']
            spec['cgf_terms'].append(entry)

        elif method == 'set_action_text':
            if args:
                spec['action_text'] = args[0] or ''

        elif method == 'set_mf_equation':
            if len(args) >= 2:
                spec['mf_equations'].append({
                    'saddle': args[0], 'rhs': args[1],
                })

        elif method == 'equation':
            # New DAE form: ``.equation(lhs=..., rhs=...,
            # population=...)``.  All three are kwargs in the canonical
            # emission, but allow positional-only too.
            lhs = kw.get('lhs') if 'lhs' in kw else (args[0] if len(args) >= 1 else '')
            rhs = kw.get('rhs') if 'rhs' in kw else (args[1] if len(args) >= 2 else '')
            pop = (kw.get('population') if 'population' in kw
                   else (args[2] if len(args) >= 3 else None))
            spec['equations'].append({
                'lhs':        lhs or '',
                'rhs':        rhs or '',
                'population': pop,
            })

        elif method == 'stability_analysis':
            # ``.stability_analysis(True)`` toggle.  Absence ⇒ default
            # OFF (matches ``TheoryBuilder``'s own default).
            if args:
                spec['stability_analysis'] = bool(args[0])
            elif 'enabled' in kw:
                spec['stability_analysis'] = bool(kw['enabled'])

        elif method == 'markovianize':
            # ``.markovianize(False)`` toggle.  Absence ⇒ default ON
            # (matches ``TheoryBuilder``'s own default).
            if args:
                spec['markovianize_default'] = bool(args[0])
            elif 'enabled' in kw:
                spec['markovianize_default'] = bool(kw['enabled'])
            else:
                spec['markovianize_default'] = True

        elif method == 'spatial_dim':
            # ``.spatial_dim(d)`` bulk-set convenience.  Apply to every
            # physical field already parsed AND remember as the default
            # for fields parsed afterwards.  (Per-field ``spatial_dim=``
            # kwargs, captured in the physical_field branch above, still
            # override.)  Mirrors TheoryBuilder.spatial_dim semantics.
            if args:
                d = int(args[0])
                spec['_default_spatial_dim'] = d
                for entry in spec['physical_fields']:
                    entry.setdefault('spatial_dim', d)

        elif method == 'boundary':
            # ``.boundary(mode, **params)`` — mode positional, params
            # (e.g. length) as kwargs.
            entry = {'mode': args[0] if args else kw.get('mode', '')}
            for k, v in kw.items():
                if k != 'mode':
                    entry[k] = v
            spec['boundary'] = entry

        elif method == 'initial':
            # ``.initial(mode, **params)`` — mode positional or kwarg.
            entry = {'mode': args[0] if args else kw.get('mode', 'stationary')}
            for k, v in kw.items():
                if k != 'mode':
                    entry[k] = v
            spec['initial'] = entry

        elif method == 'build':
            continue

    # Sync n_populations with the populations list when available.
    if spec['populations']:
        spec['n_populations'] = len(spec['populations'])
    return spec


def _strip_doc_boilerplate(doc: str) -> str:
    """Remove the auto-generated header lines that
    :func:`render_theory_file` inserts so the user only sees their
    own description on a re-load."""
    if not doc:
        return ''
    # The header pattern is:  "<name> — text-driven theory file.\n\n<user>\n\n
    # Generated by the theory-input UI ...\nLoaded by ..."
    lines = doc.splitlines()
    # Drop the first line ("<name> — text-driven theory file.")
    if lines and '—' in lines[0]:
        lines = lines[1:]
    # Drop trailing "Generated by..." and "Loaded by..." lines + the
    # blank line before them.
    keep = []
    for ln in lines:
        if ln.startswith('Generated by the theory-input UI'):
            break
        if ln.startswith('Loaded by '):
            break
        keep.append(ln)
    # Trim leading / trailing blank lines.
    while keep and not keep[0].strip():
        keep.pop(0)
    while keep and not keep[-1].strip():
        keep.pop()
    return '\n'.join(keep)

"""Emit a typed diagram as ``tikz-feynman`` source.

Produces the diagram style used in the paper:

* **time flows right to left** — external (observation) points on the left,
  sources on the right, propagator arrows pointing leftward in time;
* **source and interaction vertices are solid** (``[dot]``), carrying their
  vertex factor as a label;
* **external vertices are unshaded** (``[empty dot]``), labelled
  ``\\delta\\phi(y_i)``;
* **edges are labelled** with the propagator symbol (``G`` by default).

The output is a ``tikzpicture`` body by default, or a compilable standalone
document with ``standalone=True``.  Nothing here imports Sage beyond what a
coefficient's ``_latex_`` needs, so the emitter can be unit-tested on a
hand-built diagram.
"""

from engine.diagrams.typed_diagram_layout import DX, layout_typed_diagram

__all__ = ['to_tikz_feynman', 'diagram_to_standalone',
           'DEFAULT_SYMBOL_MAP', 'edge_style_map', 'vertex_symbol_map']

_PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{tikz}
\usepackage[compat=1.1.0]{tikz-feynman}
\begin{document}
"""
_POSTAMBLE = "\\end{document}\n"


# Sage renders a bare symbol name as ``\mathit{eps}``; the paper wants
# ``\varepsilon``.  Substitutions are applied to the rendered LaTeX, longest
# name first so ``xstar`` is not clipped by a shorter key.  Extend or override
# per call with ``symbol_map=``.
DEFAULT_SYMBOL_MAP = {
    'eps':      r'\varepsilon',
    'epsilon':  r'\varepsilon',
    'mu':       r'\mu',
    'lam':      r'\lambda',
    'lambda_X': r'\lambda_X',
    'tauc':     r'\tau_c',
    'tau_g':    r'\tau_g',
    # NOTE: <field>star is handled generically in _apply_symbol_map, which
    # renders the saddle in the model's OWN field name (x -> x^{*}).  That is
    # self-consistent with the external labels (\delta x(y_1)); mapping it to
    # a fixed \phi^{*} here would contradict them on any model whose field is
    # not called phi.
}


def _apply_symbol_map(tex, symbol_map):
    """Rewrite ``\mathit{name}`` / bare ``name`` occurrences via the map.

    A generic ``<field>star -> <field>^{*}`` rule runs first.  Every model
    names its mean-field saddle ``<field>star``, so enumerating them in the
    explicit map is hopeless -- without this, a model the map has not heard
    of renders its saddle as ``\mathit{ystar}``.
    """
    if not symbol_map:
        return tex
    import re
    tex = re.sub(r'\\mathit\{([A-Za-z][A-Za-z0-9]*)star\}', r'\1^{*}', tex)
    tex = re.sub(r'(?<![A-Za-z_\\])([A-Za-z][A-Za-z0-9]*)star(?![A-Za-z0-9_])',
                 r'\1^{*}', tex)
    for name in sorted(symbol_map, key=len, reverse=True):
        repl = symbol_map[name]
        tex = tex.replace(r'\mathit{%s}' % name, repl)
        # bare occurrences, but never inside a longer identifier
        tex = re.sub(r'(?<![A-Za-z_\\])%s(?![A-Za-z0-9_])' % re.escape(name),
                     repl.replace('\\', '\\\\'), tex)
    return tex


def _latex_coefficient(obj):
    """Best-effort LaTeX for a vertex/source coefficient.

    Sage expressions expose ``_latex_``; anything else falls back to ``str``.
    A coefficient that renders as empty is dropped rather than producing an
    empty ``$$``.
    """
    if obj is None:
        return ''
    for attr in ('_latex_', 'latex'):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return str(fn()).strip()
            except Exception:
                break
    return str(obj).strip()


def _vertex_label(vtype, show_factors, symbol_map):
    if not show_factors or vtype is None:
        return ''
    return _apply_symbol_map(
        _latex_coefficient(getattr(vtype, 'coefficient', None)), symbol_map)


def _field_symbol(leg):
    """Physics symbol for a leg ``(field_base, pop_idx)``.

    Response bases carry a trailing ``t`` (``yt`` = y-tilde) and physical
    bases a leading ``d`` (``dx`` = delta-x); strip the marker so the
    subscript reads as the field name.
    """
    if not leg:
        return ''
    base = leg[0] if isinstance(leg, (tuple, list)) else str(leg)
    base = str(base)
    if base.endswith('t') and len(base) > 1:
        base = base[:-1]
    elif base.startswith('d') and len(base) > 1:
        base = base[1:]
    return base


def _edge_label(diagram, edge_key, propagator_label):
    """Per-edge propagator label.

    ``'auto'`` subscripts the propagator with the response and physical
    field of that line -- meaningless for a single-field model (every edge
    would read the same) but essential for a multi-field one, where
    different edges ARE different matrix elements of G.
    """
    if propagator_label != 'auto':
        return propagator_label
    edge_types = getattr(diagram, 'edge_types', None) or {}
    legs = edge_types.get(edge_key)
    if not legs:
        return 'G'
    resp, phys = _field_symbol(legs[0]), _field_symbol(legs[1])
    if not resp or not phys or resp == phys:
        return 'G' if not resp else 'G_{%s}' % resp
    return 'G_{%s%s}' % (resp, phys)


# Candidate compass angles for a vertex-factor label, in preference order:
# straight up/down read best, the diagonals are fallbacks.
_LABEL_ANGLES = (90, 270, 45, 135, 315, 225, 0, 180)
_LABEL_RADIUS = 0.42        # where the label sits, in layout units


def _point_seg_distance(p, a, b):
    """Distance from point ``p`` to segment ``a``-``b``."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _best_label_angle(v, pos, edges, occupied, radius=_LABEL_RADIUS):
    """Angle whose label position is furthest from anything already drawn.

    A fixed angle puts the factor wherever the rule says, and on a busy
    diagram that can be on the far side of two other propagators from the
    vertex it belongs to -- at which point the reader cannot tell WHICH
    vertex it labels.  Score each candidate by its clearance from every edge
    segment, every other vertex, and every label already placed, and take
    the best.  Ties break toward the earlier (more readable) angle.
    """
    import math as _m
    vx, vy = pos[v]
    segs = [(pos[a], pos[b]) for a, b in edges if a in pos and b in pos]
    # Other vertices count as obstacles too: a label that drifts toward a
    # neighbouring dot reads as labelling THAT vertex instead.
    others = [p for u, p in pos.items() if u != v]
    best, best_clear = _LABEL_ANGLES[0], -1.0
    for ang in _LABEL_ANGLES:
        r = _m.radians(ang)
        lx, ly = vx + radius * _m.cos(r), vy + radius * _m.sin(r)
        clear = min(
            [_point_seg_distance((lx, ly), a, b)
             for a, b in segs
             if not (a == (vx, vy) and b == (vx, vy))]
            + [((lx - ox) ** 2 + (ly - oy) ** 2) ** 0.5
               for ox, oy in others]
            + [((lx - ox) ** 2 + (ly - oy) ** 2) ** 0.5
               for ox, oy in occupied] or [9e9])
        if clear > best_clear + 1e-9:
            best, best_clear = ang, clear
    return best


# Line styles cycled across distinct (response, physical) field pairs.  All
# keep the ``fermion`` arrow, since the arrow carries causality and must not
# be traded away for the sake of distinguishing components.
_EDGE_STYLES = ('', 'dashed', 'dotted', 'dash dot', 'densely dashed',
                'loosely dotted', 'dash dot dot')


def _field_pair(diagram, edge_key):
    """``(response, physical)`` field symbols for one edge, or None."""
    legs = (getattr(diagram, 'edge_types', None) or {}).get(edge_key)
    if not legs:
        return None
    return (_field_symbol(legs[0]), _field_symbol(legs[1]))


def edge_style_map(diagram):
    """Assign a distinct line style to each field pairing present.

    In a multi-field model different edges are different components of G, and
    a label alone makes the reader parse subscripts to see the structure.
    Giving each pairing its own stroke makes it visible at a glance.  The
    mapping is sorted, so the same pairing gets the same style in every panel
    of a figure.
    """
    et = getattr(diagram, 'edge_types', None) or {}
    pairs = sorted({(_field_symbol(a), _field_symbol(b))
                    for a, b in et.values()})
    return {p: _EDGE_STYLES[i % len(_EDGE_STYLES)]
            for i, p in enumerate(pairs)}


try:                                    # pragma: no cover - import guard
    from engine.core.vertices import SourceType as _SourceType
except Exception:                       # pragma: no cover
    _SourceType = ()


def _is_source(vtype):
    """True for a SOURCE (noise cumulant) vertex, False for an interaction.

    A source has only response legs -- bigrade ``(n, 0)``.  ``isinstance``
    settles it for the real types; the bigrade fallback keeps hand-built
    stand-ins working.  Anything carrying no evidence either way is an
    interaction, so a bare object with just a ``coefficient`` still gets a
    ``v_j``.
    """
    if _SourceType and isinstance(vtype, _SourceType):
        return True
    bg = getattr(vtype, 'bigrade', None)
    if bg is not None and len(tuple(bg)) == 2:
        n_resp, n_phys = tuple(bg)
        return n_phys == 0 and n_resp > 0
    return False


def _source_n_legs(vtype):
    """Number of legs on a source -- the order of the noise cumulant."""
    legs = getattr(vtype, 'response_legs', None)
    if legs is not None:
        return len(legs)
    bg = getattr(vtype, 'bigrade', None)
    return int(tuple(bg)[0]) if bg else 0


def _source_field(vtype):
    """The field a source's legs belong to, or '' if they are not all one."""
    legs = getattr(vtype, 'response_legs', None) or []
    syms = {_field_symbol(l) for l in legs}
    return syms.pop() if len(syms) == 1 else ''


def _factor_key(vtype, tex):
    """Figure-wide identity of a vertex factor.

    Two factors are the same entry in the key only if they are the same KIND
    (source vs interaction) with the same expression -- and, for a source,
    the same number of legs, since that is what its symbol is built from.
    """
    if _is_source(vtype):
        return ('source', _source_n_legs(vtype), tex)
    return ('vertex', tex)


def _symbol_for(vsym, vtype, tex):
    """Look a factor up in a symbol map keyed either way.

    ``vertex_symbol_map(..., keyed=True)`` keys by ``_factor_key`` so that a
    source and an interaction that happen to print the same expression stay
    distinct; a plain ``{tex: symbol}`` dict (what callers passed before, and
    what the key table is built from) still resolves.
    """
    if not vsym:
        return tex
    key = _factor_key(vtype, tex)
    if key in vsym:
        return vsym[key]
    return vsym.get(tex, tex)


def vertex_symbol_map(diagrams, symbol_map=None, keyed=False):
    """Short symbol for each distinct vertex factor, plus its LaTeX.

    Printing ``3\,\varepsilon x^{*}_{1}`` on a vertex is unreadable once a
    diagram has more than a couple of them, and at two loops the copies
    collide outright.  Naming the factors ``v_1, v_2, ...`` and giving the
    expressions once in a key keeps every value while freeing the drawing.

    Accepts one diagram or MANY.  Pass the whole set: numbering built
    per-diagram would make ``v_1`` mean one factor in one panel and a
    different factor in the next, and the key would list only whatever the
    first panel happened to contain.

    The two kinds of vertex are named differently because they mean
    different things.  A SOURCE is a noise cumulant, and its order is the
    only thing a reader needs from the drawing, so it is named
    ``\kappa_n`` with ``n`` its number of legs -- an arbitrary index would
    hide information the diagram already carries.  An INTERACTION has no
    such canonical name, so it keeps ``v_1, v_2, ...``, numbered over the
    distinct interaction factors only (sources do not consume a ``v``).

    ``keyed=True`` keys the result by ``_factor_key`` instead of by the
    expression, which keeps a source and an interaction apart even if their
    coefficients print identically.  The default expression-keyed form is
    what the key table and older callers expect.
    """
    if symbol_map is None:
        symbol_map = DEFAULT_SYMBOL_MAP
    if not isinstance(diagrams, (list, tuple)):
        diagrams = [diagrams]
    entries, seen = [], set()
    for dia in diagrams:
        assignments = getattr(dia, 'vertex_assignments', {}) or {}
        for v in sorted(assignments):
            vtype = assignments[v]
            tex = _apply_symbol_map(
                _latex_coefficient(getattr(vtype, 'coefficient', None)),
                symbol_map)
            if not tex:
                continue
            key = _factor_key(vtype, tex)
            if key not in seen:
                seen.add(key)
                entries.append((key, tex, vtype))

    # A source's name is fixed by its leg count, so two DIFFERENT source
    # factors of the same order would both want the same name.  Only then --
    # and only over the colliding group -- add a superscript, so the common
    # single-noise case stays plain \kappa_2.
    extra = {}
    by_order = {}
    for key, tex, vtype in entries:
        if key[0] == 'source':
            by_order.setdefault(key[1], []).append((key, vtype))
    for _n, group in by_order.items():
        if len(group) < 2:
            continue
        fields = [_source_field(vt) for _k, vt in group]
        distinct_fields = all(fields) and len(set(fields)) == len(group)
        for i, (key, _vt) in enumerate(group):
            extra[key] = ('^{%s}' % fields[i] if distinct_fields
                          else '^{(%d)}' % (i + 1))

    out, n_int = {}, 0
    for key, tex, _vtype in entries:
        if key[0] == 'source':
            sym = r'\kappa_{%d}%s' % (key[1], extra.get(key, ''))
        else:
            n_int += 1
            sym = r'v_{%d}' % n_int
        out[key if keyed else tex] = sym
    return out


def _node_name(v):
    """TikZ node names must be simple; vertex ids are ints."""
    return 'v%s' % v


def to_tikz_feynman(diagram, *, propagator_label='G',
                    external_label='auto',
                    show_factors='auto', symbol_map=None, scale=1.0,
                    factor_label_angle='auto', dot_size=None, bend_angle=14,
                    external_label_distance='1pt', style_by_field=False,
                    symbolic_factors=False, vertex_symbols=None,
                    indent='  '):
    """Return ``tikz-feynman`` source for one typed diagram or prediagram.

    Parameters
    ----------
    diagram : TypedDiagram or (D, G, leaves, internal)
        A typed diagram draws vertex factors and knows sources from
        interactions; a bare prediagram still draws, with generic styling.
    propagator_label : str or None
        Edge label (``None`` suppresses it).  ``'auto'`` subscripts each
        propagator with its response and physical field, which distinguishes
        the matrix elements of G in a multi-field model, and names each
        distinct component ONCE rather than on every edge carrying it.
    external_label : str
        ``'auto'`` (default) names each external by its own field, e.g.
        ``\delta x(y_1)`` -- correct for a multi-field model where the
        externals are not all the same field.  Otherwise a ``%d``-style
        template receiving the 1-based external index.
    show_factors : bool or 'auto'
        ``'auto'`` (default) prints each DISTINCT coefficient once; ``True``
        prints one on every vertex, which past one loop repeats a single value
        until the copies collide; ``False`` suppresses them.
    symbol_map : dict or None
        Parameter-name -> LaTeX overrides applied to vertex factors.
        ``None`` uses :data:`DEFAULT_SYMBOL_MAP`; pass ``{}`` to disable.
    factor_label_angle : int or 'auto'
        Compass angle (degrees) for the vertex-factor label; 90 = above.
        ``'auto'`` (default) labels upward for vertices at or above the
        midline and downward for those below, spreading the factors instead
        of stacking them into one band.
        Vertex factors go ABOVE and edge labels BELOW by default -- the two
        label families are separated by convention, because placing both near
        a vertex makes them overlap on any diagram with fanned-out edges.
    dot_size : str or None
        Overrides the filled-vertex diameter, e.g. ``'1.6mm'``.  The
        tikz-feynman default is large relative to our column pitch.
    bend_angle : int
        Degrees of bow applied to each side of a parallel-edge bundle.  Kept
        small: enough to separate the two lines of a bubble, not enough to
        make a straight propagator look curved.
    external_label_distance : str
        Gap between an external node and its label.
    style_by_field : bool
        Give each distinct (response, physical) field pairing its own line
        style, so the component structure of a multi-field diagram is visible
        without reading subscripts.  All styles keep the causal arrow.  When
        set, edge labels are redundant and should be turned off; the key
        emitted by ``tikz_figure.legend_table`` carries the mapping.
    symbolic_factors : bool
        Label vertices ``v_1, v_2, ...`` instead of printing the coefficient,
        with the expressions given once in the key.  Short names are cheap
        enough to repeat on every vertex; expressions are not.
    vertex_symbols : dict or None
        A symbol map shared across a whole figure.  Supply this whenever
        drawing more than one panel, or each panel numbers its own factors
        and the same symbol means different things in different panels.
    scale : float
        ``tikzpicture`` scale factor.

    Returns
    -------
    str
        A ``\\begin{tikzpicture}`` ... ``\\end{tikzpicture}`` block.
    """
    prediagram = getattr(diagram, 'prediagram', diagram)
    D, _G, leaves, _internal = prediagram
    leaf_set = set(leaves)
    assignments = getattr(diagram, 'vertex_assignments', {}) or {}
    if symbol_map is None:
        symbol_map = DEFAULT_SYMBOL_MAP
    pos = layout_typed_diagram(diagram)

    lines = []
    pic_opts = ['scale=%g' % scale]
    if dot_size:
        pic_opts.append(
            r'/tikzfeynman/every dot/.style={minimum size=%s}' % dot_size)
        pic_opts.append(
            r'/tikzfeynman/every empty dot/.style={minimum size=%s}' % dot_size)
    lines.append(r'\begin{tikzpicture}[%s]' % ', '.join(pic_opts))
    lines.append(indent + r'\begin{feynman}')

    # ── vertices ────────────────────────────────────────────────────
    edge_list = sorted(D.edges(labels=False))
    style_of = edge_style_map(diagram) if style_by_field else {}
    # Each distinct propagator symbol is named once.  Repeating one symbol on
    # every edge tells the reader nothing after the first time, and once the
    # line STYLE also encodes the component the labels are pure clutter.
    _seen_edge_labels = set()
    placed_labels = []
    ext_legs = getattr(diagram, 'external_legs', None) or {}
    for idx, v in enumerate(sorted(leaves), start=1):
        x, y = pos[v]
        if external_label == 'auto':
            sym = _field_symbol(ext_legs.get(v)) or r'\phi'
            lab = r'\delta %s(y_{%d})' % (sym, idx)
        else:
            lab = external_label % idx
        # The label rides on a ``label=`` option, NOT in the node body: an
        # ``[empty dot]`` sized to contain ``\delta x(y_1)`` becomes a huge
        # circle with text inside it.  Placed at 180 degrees it sits to the
        # LEFT of a small circle, outside the diagram, where nothing crosses it.
        lines.append(
            indent * 2 + r'\vertex [empty dot, label={[label distance=%s]'
                         r'180:\(%s\)}] (%s) at (%.3f, %.3f) {};'
            % (external_label_distance, lab, _node_name(v), x, y))

    # ``show_factors='auto'``: print each DISTINCT factor once.  Every
    # interaction vertex of a one-species model carries the same coefficient,
    # so labelling all of them repeats one value as many times as the loop
    # order allows and, past one loop, the copies collide with each other.
    # One instance of each distinct value loses no information.
    _seen_factors = set()
    vsym = (vertex_symbols if vertex_symbols is not None
            else (vertex_symbol_map(diagram, symbol_map, keyed=True)
                  if symbolic_factors else {}))
    for v in sorted(set(D.vertices()) - leaf_set):
        x, y = pos[v]
        label = _vertex_label(assignments.get(v),
                              bool(show_factors), symbol_map)
        if label and symbolic_factors:
            # Symbolic mode names EVERY vertex -- the whole point is that the
            # short name is cheap enough to repeat, unlike the expression.
            label = _symbol_for(vsym, assignments.get(v), label)
        elif label and show_factors == 'auto':
            if label in _seen_factors:
                label = ''
            else:
                _seen_factors.add(label)
        # A ``[dot]`` vertex is a FILLED node: anything in its node body is
        # drawn inside the ink and invisible.  The factor must therefore ride
        # on a ``label=`` option, placed below the vertex so it clears the
        # propagator lines converging on it.
        opts = 'dot'
        if label:
            # Place the factor on the side AWAY from the diagram's midline:
            # vertices above y=0 label upward, below label downward.  Pinning
            # every factor to one side stacks them into the same horizontal
            # band as the edge labels, which is what makes a dense panel
            # unreadable.  An explicit angle overrides the heuristic.
            if factor_label_angle != 'auto':
                angle = factor_label_angle
            else:
                angle = _best_label_angle(v, pos, edge_list, placed_labels)
                import math as _m
                _r = _m.radians(angle)
                placed_labels.append(
                    (x + _LABEL_RADIUS * _m.cos(_r),
                     y + _LABEL_RADIUS * _m.sin(_r)))
            opts += r', label={[label distance=1pt]%d:\(%s\)}' % (
                angle, label)
        lines.append(indent * 2 + r'\vertex [%s] (%s) at (%.3f, %.3f) {};'
                     % (opts, _node_name(v), x, y))

    # ── edges ───────────────────────────────────────────────────────
    # Edge direction is the retarded propagator's time arrow (earlier ->
    # later).  Since later times sit further LEFT, drawing tail -> head
    # already points the arrow leftward; no reversal is needed.

    lines.append(indent * 2 + r'\diagram* {')
    edges = sorted(D.edges(labels=False))
    # Parallel edges (a loop between two vertices) must be bowed apart or
    # tikz-feynman draws them on top of each other.  Bend symmetrically about
    # the straight line so the pair reads as a bubble.
    from collections import Counter
    multiplicity = Counter(edges)
    drawn = Counter()
    for i, (u, v) in enumerate(edges):
        n = multiplicity[(u, v)]
        j = drawn[(u, v)]
        drawn[(u, v)] += 1
        bend = ''
        if n > 1:
            # Bow parallel propagators apart symmetrically.  NOT ``half
            # left/right``: that is a SEMICIRCULAR arc, so between two distant
            # vertices it balloons into a circle bigger than the diagram.  A
            # fixed modest angle bows by the same visual amount at any
            # separation.
            offset = j - (n - 1) / 2.0
            if abs(offset) > 1e-9:
                side = 'left' if offset < 0 else 'right'
                bend = ', bend %s=%d' % (
                    side, int(round(bend_angle * abs(offset) * 2)))
        # Per-edge label: with ``propagator_label='auto'`` different lines
        # are different matrix elements of G and must be named separately.
        lab = _edge_label(diagram, (u, v, j), propagator_label)
        if lab and propagator_label == 'auto':
            if lab in _seen_edge_labels:
                lab = None
            else:
                _seen_edge_labels.add(lab)
        opts = ['fermion']
        if style_of:
            st = style_of.get(_field_pair(diagram, (u, v, j)))
            if st:
                opts.append(st)
        if lab:
            # ``edge label'`` (primed) sits on the far side of the line from
            # ``edge label``, keeping it clear of the vertex factors above.
            opts.append(r"edge label'=\(%s\)" % lab)
        sep = ',' if i < len(edges) - 1 else ''
        lines.append(indent * 3 + '(%s) -- [%s%s] (%s)%s'
                     % (_node_name(u), ', '.join(opts), bend,
                        _node_name(v), sep))
    lines.append(indent * 2 + '};')

    lines.append(indent + r'\end{feynman}')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines) + '\n'


def diagram_to_standalone(diagram, **kw):
    """``to_tikz_feynman`` wrapped in a compilable standalone document."""
    return _PREAMBLE + to_tikz_feynman(diagram, **kw) + _POSTAMBLE

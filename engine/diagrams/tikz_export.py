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

__all__ = ['to_tikz_feynman', 'diagram_to_standalone', 'DEFAULT_SYMBOL_MAP']

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


def _node_name(v):
    """TikZ node names must be simple; vertex ids are ints."""
    return 'v%s' % v


def to_tikz_feynman(diagram, *, propagator_label='G',
                    external_label='auto',
                    show_factors=True, symbol_map=None, scale=1.0,
                    factor_label_angle='auto', dot_size=None, bend_angle=28,
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
        the matrix elements of G in a multi-field model.
    external_label : str
        ``'auto'`` (default) names each external by its own field, e.g.
        ``\delta x(y_1)`` -- correct for a multi-field model where the
        externals are not all the same field.  Otherwise a ``%d``-style
        template receiving the 1-based external index.
    show_factors : bool
        Print each interaction/source vertex's coefficient on the vertex.
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
        Degrees of bow applied to each side of a parallel-edge bundle.
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
    ext_legs = getattr(diagram, 'external_legs', None) or {}
    for idx, v in enumerate(sorted(leaves), start=1):
        x, y = pos[v]
        if external_label == 'auto':
            sym = _field_symbol(ext_legs.get(v)) or r'\phi'
            lab = r'\delta %s(y_{%d})' % (sym, idx)
        else:
            lab = external_label % idx
        lines.append(
            indent * 2 + r'\vertex [empty dot] (%s) at (%.3f, %.3f) {\(%s\)};'
            % (_node_name(v), x, y, lab))

    for v in sorted(set(D.vertices()) - leaf_set):
        x, y = pos[v]
        label = _vertex_label(assignments.get(v), show_factors, symbol_map)
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
            elif abs(y) > 1e-9:
                angle = 90 if y > 0 else 270
            else:
                # Every vertex of a linear chain sits ON the midline, so the
                # away-from-midline rule would put all their factors in one
                # band again.  Alternate by column instead.
                angle = 90 if int(round(-x / DX)) % 2 == 0 else 270
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
        opts = ['fermion']
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

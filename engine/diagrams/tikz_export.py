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

from engine.diagrams.typed_diagram_layout import layout_typed_diagram

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
    'xstar':    r'\phi^{*}',
    'nstar':    r'n^{*}',
    'vstar':    r'v^{*}',
}


def _apply_symbol_map(tex, symbol_map):
    """Rewrite ``\mathit{name}`` / bare ``name`` occurrences via the map."""
    if not symbol_map:
        return tex
    import re
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


def _node_name(v):
    """TikZ node names must be simple; vertex ids are ints."""
    return 'v%s' % v


def to_tikz_feynman(diagram, *, propagator_label='G',
                    external_label=r'\delta\phi(y_{%d})',
                    show_factors=True, symbol_map=None, scale=1.0,
                    indent='  '):
    """Return ``tikz-feynman`` source for one typed diagram or prediagram.

    Parameters
    ----------
    diagram : TypedDiagram or (D, G, leaves, internal)
        A typed diagram draws vertex factors and knows sources from
        interactions; a bare prediagram still draws, with generic styling.
    propagator_label : str or None
        Edge label (``None`` suppresses it).
    external_label : str
        ``%d``-style template receiving the 1-based external index.
    show_factors : bool
        Print each interaction/source vertex's coefficient on the vertex.
    symbol_map : dict or None
        Parameter-name -> LaTeX overrides applied to vertex factors.
        ``None`` uses :data:`DEFAULT_SYMBOL_MAP`; pass ``{}`` to disable.
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
    lines.append(r'\begin{tikzpicture}[scale=%g]' % scale)
    lines.append(indent + r'\begin{feynman}')

    # ── vertices ────────────────────────────────────────────────────
    for idx, v in enumerate(sorted(leaves), start=1):
        x, y = pos[v]
        lines.append(
            indent * 2 + r'\vertex [empty dot] (%s) at (%.3f, %.3f) {\(%s\)};'
            % (_node_name(v), x, y, external_label % idx))

    for v in sorted(set(D.vertices()) - leaf_set):
        x, y = pos[v]
        label = _vertex_label(assignments.get(v), show_factors, symbol_map)
        body = r' {\(%s\)}' % label if label else ''
        lines.append(indent * 2 + r'\vertex [dot] (%s) at (%.3f, %.3f)%s;'
                     % (_node_name(v), x, y, body))

    # ── edges ───────────────────────────────────────────────────────
    # Edge direction is the retarded propagator's time arrow (earlier ->
    # later).  Since later times sit further LEFT, drawing tail -> head
    # already points the arrow leftward; no reversal is needed.
    opts = ['fermion']
    if propagator_label:
        opts.append(r'edge label=\(%s\)' % propagator_label)
    opt_str = ', '.join(opts)

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
            # n=2 -> (left, right); n=3 -> (left, straight, right); etc.
            amount = 50 if n == 2 else 60
            offset = j - (n - 1) / 2.0
            if abs(offset) > 1e-9:
                side = 'left' if offset < 0 else 'right'
                bend = ', half %s' % side if n == 2 else \
                       ', bend %s=%d' % (side, int(amount * abs(offset) * 2))
        sep = ',' if i < len(edges) - 1 else ''
        lines.append(indent * 3 + '(%s) -- [%s%s] (%s)%s'
                     % (_node_name(u), opt_str, bend, _node_name(v), sep))
    lines.append(indent * 2 + '};')

    lines.append(indent + r'\end{feynman}')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines) + '\n'


def diagram_to_standalone(diagram, **kw):
    """``to_tikz_feynman`` wrapped in a compilable standalone document."""
    return _PREAMBLE + to_tikz_feynman(diagram, **kw) + _POSTAMBLE

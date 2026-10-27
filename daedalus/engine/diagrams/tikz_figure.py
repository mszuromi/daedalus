"""Assemble many diagrams into one figure.

``tikz_export`` draws a single diagram; a paper needs the whole set at a given
``(k, ell)`` laid out as an array, with each panel captioned by its loop order
and multiplicity.  This module does that assembly and nothing else, so the
per-diagram emitter stays simple.

Two output shapes:

* :func:`diagrams_to_figure` -- a ``figure`` body of ``subcaptionbox`` panels,
  ready to paste into a manuscript;
* :func:`diagrams_to_standalone` -- the same wrapped in a compilable document,
  for checking the array before it goes near the paper.
"""

from engine.diagrams.tikz_export import (
    to_tikz_feynman, edge_style_map, vertex_symbol_map,
)
from engine.diagrams.typed_diagram_layout import DX, layout_typed_diagram

__all__ = ['diagrams_to_figure', 'diagrams_to_standalone',
           'legend_table', 'diagrams_to_pages', 'PREAMBLE_PACKAGES']

PREAMBLE_PACKAGES = r"""\usepackage{tikz}
\usepackage[compat=1.1.0]{tikz-feynman}
\usepackage{subcaption}
\usepackage{amsmath}"""

_STANDALONE_HEAD = r"""\documentclass[border=10pt]{standalone}
%s
\begin{document}
\begin{minipage}{%s}
"""
_STANDALONE_TAIL = r"""\end{minipage}
\end{document}
"""


def _panel_caption(rec, index):
    """Caption for one panel: loop order and, if known, the multiplicity."""
    bits = []
    ell = rec.get('ell') if isinstance(rec, dict) else None
    if ell is not None:
        bits.append(r'$\ell=%d$' % ell)
    mult = None
    if isinstance(rec, dict):
        for key in ('multiplicity', 'combined_prefactor', 'M'):
            if rec.get(key) is not None:
                mult = rec[key]
                break
    if mult is not None:
        try:
            bits.append(r'$M=%s$' % _fmt_number(mult))
        except Exception:
            pass
    return ', '.join(bits) if bits else '(%d)' % index


def _fmt_number(x):
    """Render a multiplicity compactly: integers bare, else 4 significant digits."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(f - round(f)) < 1e-12:
        return '%d' % round(f)
    return '%.4g' % f


def _auto_scale(diagram, ncol, budget_cols=2.0):
    """Shrink a panel so a wide diagram still fits its column.

    A high-k or high-loop diagram can be five or more layout columns wide;
    drawn at the same scale as a tree it overruns its panel and collides with
    the neighbour.  Scale inversely with the diagram's own width, measured in
    columns, and never enlarge beyond the caller's scale.

    ``budget_cols=2.0`` is set from the geometry, not taste: the default
    panel is 0.30\textwidth, so at a 16 cm text width each panel gets
    ~4.8 cm, and a column is DX = 2.3 units ~ 2.3 cm at scale 1.  Widen
    ``panel_width`` and this can rise.
    """
    try:
        pos = layout_typed_diagram(diagram)
    except Exception:
        return None
    if not pos:
        return None
    span = (max(x for x, _ in pos.values()) -
            min(x for x, _ in pos.values())) / DX
    if span <= budget_cols:
        return None
    return round(budget_cols / span, 3)


def _as_diagram(rec):
    """Accept a diagram record dict, a TypedDiagram, or a raw prediagram."""
    if isinstance(rec, dict):
        return rec.get('typed_diagram') or rec.get('prediagram') or rec
    return rec


def diagrams_to_figure(records, *, ncol=3, panel_width=r'0.30\textwidth',
                       scale=0.75, caption=None, label=None,
                       max_panels=None, propagator_label=None, **tikz_kw):
    """Lay diagrams out as a grid of ``subcaptionbox`` panels.

    Parameters
    ----------
    records : iterable
        Diagram records (dicts with ``typed_diagram``/``ell``), TypedDiagrams,
        or raw prediagram tuples -- mixed input is fine.
    ncol : int
        Panels per row.
    panel_width : str
        LaTeX width for each panel.
    scale : float or 'auto'
        ``tikzpicture`` scale inside each panel.  A wide diagram is shrunk
        below this automatically so it does not overrun its column and
        collide with the neighbouring panel.
    caption, label : str or None
        Figure-level caption and ``\\label``; omitted when ``None`` so the
        result can be embedded in a caller's own float.
    propagator_label : str or None
        Defaults to ``None`` for an ARRAY, unlike a single diagram which
        defaults to ``'G'``.  A lone diagram is pedagogical and wants its
        propagators named; an array is about structure, and repeating one
        symbol on every edge of every panel adds no information while
        crowding the vertex factors.  Pass ``'G'`` for a small array.
    max_panels : int or None
        Draw at most this many, appending a note recording how many were
        dropped.  Silent truncation would misrepresent the diagram set, so
        the note is always emitted when it bites.

    Returns
    -------
    str
    """
    records = list(records)
    total = len(records)
    dropped = 0
    if max_panels is not None and total > max_panels:
        dropped = total - max_panels
        records = records[:max_panels]

    # ONE symbol map for the whole array, for the same reason as
    # ``diagrams_to_pages``: built per panel, ``v_1`` would name a different
    # factor in each one.  Built over the panels actually drawn, so the key
    # matches the figure after truncation.
    if tikz_kw.get('symbolic_factors') and 'vertex_symbols' not in tikz_kw:
        tikz_kw['vertex_symbols'] = vertex_symbol_map(
            [_as_diagram(r) for r in records], tikz_kw.get('symbol_map'),
            keyed=True)
    # Line styles are a symbol map too, and go wrong the same way: handed out
    # per panel, one stroke names a different component in each.
    if tikz_kw.get('style_by_field') and 'edge_styles' not in tikz_kw:
        tikz_kw['edge_styles'] = edge_style_map(
            [_as_diagram(r) for r in records])
    out = [r'\begin{figure}[t]', r'  \centering']
    for i, rec in enumerate(records):
        dia = _as_diagram(rec)
        panel_scale = scale
        if scale == 'auto':
            panel_scale = _auto_scale(dia, ncol) or 0.75
        else:
            shrink = _auto_scale(dia, ncol)
            if shrink is not None:
                panel_scale = min(scale, shrink)
        body = to_tikz_feynman(dia, scale=panel_scale,
                               propagator_label=propagator_label, **tikz_kw)
        body = '\n'.join('      ' + ln for ln in body.strip().splitlines())
        out.append(r'  \subcaptionbox{%s}[%s]{%%'
                   % (_panel_caption(rec, i + 1), panel_width))
        # ``\subcaptionbox`` neither clips nor scales its content: a picture
        # wider than the panel simply overruns and collides with the next
        # one.  Shrink-to-fit makes the declared panel width authoritative.
        # The \ifdim guard keeps it shrink-ONLY, so a small diagram is not
        # blown up to fill the column.
        out.append(r'    \resizebox{\ifdim\width>\linewidth'
                   r'\linewidth\else\width\fi}{!}{%')
        out.append(body)
        out.append('    }%')
        out.append('  }')
        if (i + 1) % ncol == 0 and i + 1 < len(records):
            out.append(r'  \\[1.2em]')
    if dropped:
        out.append(r'  \\[0.8em]')
        out.append(r'  \emph{(%d of %d diagrams shown; %d omitted.)}'
                   % (len(records), total, dropped))
    if caption:
        out.append(r'  \caption{%s}' % caption)
    if label:
        out.append(r'  \label{%s}' % label)
    out.append(r'\end{figure}')
    return '\n'.join(out) + '\n'


def diagrams_to_standalone(records, *, total_width='16cm', **kw):
    """A compilable document containing the panel array.

    ``standalone`` has no ``\\textwidth``, so the panels are wrapped in a
    fixed-width ``minipage`` and relative panel widths resolve against that.
    """
    kw.setdefault('caption', None)
    kw.setdefault('label', None)
    body = diagrams_to_figure(records, **kw)
    # ``figure`` is not available in standalone.  Downgrading to ``center``
    # alone breaks ``\subcaptionbox``, which refuses to run outside a float
    # ("\setcaptionsubtype outside float"); ``\captionsetup{type=figure}``
    # supplies the float context the caption package is looking for.
    body = (body.replace(r'\begin{figure}[t]',
                         '\\begin{center}\n  \\captionsetup{type=figure}')
                .replace(r'\end{figure}', r'\end{center}'))
    return (_STANDALONE_HEAD % (PREAMBLE_PACKAGES, total_width)
            + body + _STANDALONE_TAIL)


def _style_sample(style):
    """A short line drawn in ``style``, for the key."""
    opt = 'fermion' + (', ' + style if style else '')
    return (r'\tikz[baseline=-0.5ex]{\begin{feynman}'
            r'\vertex (a) at (0,0); \vertex (b) at (1.1,0);'
            r'\diagram*{(b) -- [%s] (a)};\end{feynman}}' % opt)


def legend_table(diagram, *, edges=True, vertices=True, symbol_map=None,
                 vertex_symbols=None, edge_styles=None):
    """A key mapping line styles to field pairings and symbols to factors.

    Once the drawing encodes components by stroke and vertices by short name,
    the figure is only readable with a key.  Emitting it from the same maps
    the renderer uses means the two cannot drift apart -- which is why both
    maps are passed IN when the caller has one for the whole figure: derived
    from ``diagram`` alone, the key lists only the components and factors
    that one panel contains, and silently mis-names the rest.
    """
    rows = []
    if edges:
        styles = (edge_styles if edge_styles is not None
                  else edge_style_map(diagram))
        for (a, b), st in sorted(styles.items()):
            name = ('G_{%s%s}' % (a, b)) if a != b else ('G_{%s}' % a)
            rows.append((_style_sample(st), r'\(%s\)' % name))
    if vertices:
        vs = (vertex_symbols if vertex_symbols is not None
              else vertex_symbol_map(diagram, symbol_map, keyed=True))
        # The map may be keyed by expression (older callers, and the plain
        # form) or by factor key (kind, order, expression); the expression is
        # the last component either way.  Interactions are listed before the
        # noise cumulants so the key reads as two blocks rather than as the
        # order the panels happened to walk the vertices in.
        pairs = [((k[-1] if isinstance(k, tuple) else k), sym)
                 for k, sym in vs.items()]
        inter = [p for p in pairs if not p[1].lstrip().startswith(r'\kappa')]
        srcs = [p for p in pairs if p[1].lstrip().startswith(r'\kappa')]
        for tex, sym in inter + srcs:
            rows.append((r'\(%s\)' % sym, r'\(%s\)' % tex))
    if not rows:
        return ''
    out = [r'\begin{tabular}{@{}ll@{}}', r'  \hline']
    for left, right in rows:
        out.append('  %s & %s \\\\' % (left, right))
    out.append(r'  \hline')
    out.append(r'\end{tabular}')
    return '\n'.join(out) + '\n'


def diagrams_to_pages(records, *, ncol=2, panel_width=r'0.46\textwidth',
                      scale='auto', caption=None, label=None,
                      legend_from=None, row_skip='2.2em', **tikz_kw):
    """Every diagram, flowing across as many pages as it needs.

    ``diagrams_to_figure`` builds a ``figure``, which LaTeX floats and cannot
    break: past a page of panels it silently overflows or is dropped.  For a
    complete set -- and a complete set is the point, since a truncated one
    misrepresents the expansion -- the panels must be ordinary page content
    that flows.  The caption is set with ``\\captionof`` so it still reads as
    a figure caption without a float to carry it.
    """
    records = list(records)
    # ONE symbol map for the whole figure: numbering per panel would make the
    # same symbol mean different factors in different panels, and the key
    # would only list what the first panel happened to use.
    if tikz_kw.get('symbolic_factors') and 'vertex_symbols' not in tikz_kw:
        tikz_kw['vertex_symbols'] = vertex_symbol_map(
            [_as_diagram(r) for r in records], tikz_kw.get('symbol_map'),
            keyed=True)
    # Line styles are a symbol map too, and go wrong the same way: handed out
    # per panel, one stroke names a different component in each.
    if tikz_kw.get('style_by_field') and 'edge_styles' not in tikz_kw:
        tikz_kw['edge_styles'] = edge_style_map(
            [_as_diagram(r) for r in records])
    out = []
    if caption:
        out.append(r'\captionof{figure}{%s}' % caption)
        if label:
            out.append(r'\label{%s}' % label)
        out.append('')
    if legend_from is not None:
        key = legend_table(legend_from, symbol_map=tikz_kw.get('symbol_map'),
                           vertex_symbols=tikz_kw.get('vertex_symbols'),
                           edge_styles=tikz_kw.get('edge_styles'))
        if key:
            out.append(r'\begin{center}')
            out.append(key)
            out.append(r'\end{center}')
            out.append(r'\vspace{1em}')
    for i, rec in enumerate(records):
        dia = _as_diagram(rec)
        panel_scale = scale
        if scale == 'auto':
            panel_scale = _auto_scale(dia, ncol) or 0.9
        body = to_tikz_feynman(dia, scale=panel_scale, **tikz_kw)
        body = '\n'.join('    ' + ln for ln in body.strip().splitlines())
        out.append(r'\begin{minipage}[t]{%s}' % panel_width)
        out.append(r'  \centering')
        out.append(r'  \resizebox{\ifdim\width>\linewidth'
                   r'\linewidth\else\width\fi}{!}{%')
        out.append(body)
        out.append('  }%')
        out.append(r'  \\[0.4em] \footnotesize (%d)%s' % (
            i + 1, (r' \(\ell=%d\)' % rec['ell'])
            if isinstance(rec, dict) and rec.get('ell') is not None else ''))
        out.append(r'\end{minipage}')
        if (i + 1) % ncol == 0:
            out.append(r'\par\vspace{%s}' % row_skip)
        else:
            out.append(r'\hfill')
    return '\n'.join(out) + '\n'

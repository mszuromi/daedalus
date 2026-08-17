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

from engine.diagrams.typed_diagram_layout import (
    DX, ARROWHEAD_MM, DOT_RADIUS_MM, layout_typed_diagram, mm_per_unit,
    _seg_distance, bubble_bend_deg, BOW as _BOW, MAX_BEND as _MAX_BEND,
    MIN_BUBBLE_MM,
)

__all__ = ['to_tikz_feynman', 'diagram_to_standalone',
           'DEFAULT_SYMBOL_MAP', 'edge_style_map', 'vertex_symbol_map',
           'edge_bends', 'path_point', 'arrow_positions']

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


# ── labels, placed jointly ──────────────────────────────────────────
# A label is NOT scaled with the picture: `scale=` transforms coordinates,
# not node text, so at the panel scale used in a figure array a `v_2` prints
# 3.5 mm wide inside a diagram only 46 mm wide.  Placement therefore has to
# reason about the label's BOX in printed millimetres; scoring the clearance
# of its anchor POINT, as this first did, calls a label clear when two thirds
# of it lies across a propagator.

# Candidate compass angles, in preference order: straight up/down read best,
# the diagonals are fallbacks.
# The eight compass points FIRST -- `rank` breaks ties by this order, so a
# panel with room keeps the canonical placement it always had.  The
# half-steps are an escape for dense panels: with 45-degree steps alone a
# label whose every compass direction is blocked has to settle for lying
# ACROSS a propagator, which is what panel 10 of the two-loop figure did.
_LABEL_ANGLES = (90, 270, 45, 135, 315, 225, 0, 180,
                 68, 112, 248, 292, 22, 158, 202, 338)
# Candidate gaps between vertex and label, in TeX points.
_LABEL_DISTANCES_PT = (1.0, 4.0, 8.0)
_PT_MM = 0.35146            # 1 TeX point in millimetres
# Widths of an 11pt math atom, measured with \settowidth (see the commit).
_ATOM_MM = 2.10             # a letter, italic, incl. side bearing
_DIGIT_MM = 1.92
_PAREN_MM = 1.25
_SCRIPT = 0.72              # sub/superscripts are set smaller
_LABEL_MARGIN_MM = 0.4      # labels that merely touch still read as one
_OWNERSHIP = 1.15           # a label must stay this much nearer ITS vertex
# TikZ pads every node by `inner sep` (0.3333em, = 3.65pt at 11pt) plus
# `outer sep` (half the line width) BEFORE it works out where the node's
# border is -- and it is the border, not the text, that `label distance`
# is measured from.  Ignoring this puts the printed glyphs further out than
# the placement scored them: measured against \pgfpointanchor over the 66
# two-loop panels, +1.0 mm at a cardinal angle and +2.0 mm at a diagonal.
_INNER_SEP_MM = 1.283 + 0.07
_LINE_MM = 1.66             # height of an x-height math box
_DEEP_MM = 0.58             # extra depth once a subscript is present
_TALL_MM = 1.30             # extra height for parentheses and \delta


def _distance_pt(dim):
    """A TeX dimension string as points, for geometry (``'1pt'`` -> 1.0)."""
    import re
    m = re.match(r'\s*([-\d.]+)\s*([a-z]*)', str(dim))
    if not m:
        return 1.0
    val = float(m.group(1))
    return {'pt': 1.0, 'mm': 1.0 / _PT_MM, 'cm': 10.0 / _PT_MM,
            'ex': 4.3, 'em': 10.0}.get(m.group(2), 1.0) * val


def _label_extent_mm(tex):
    """Printed ``(width, height)`` of a math label at 11pt, in millimetres.

    Calibrated against ``\settowidth`` on the labels this emitter actually
    produces: ``v_2`` 3.54 x 2.23 mm, ``\kappa_2`` 3.89 x 2.23 mm,
    ``\delta x(y_1)`` 10.60 x 3.85 mm.  Approximate by construction -- a real
    metric needs TeX -- so callers pad it.
    """
    w, i, sub, tall = 0.0, 0, False, False
    script, group = 1.0, 0            # script scale, and its brace depth
    depth = 0
    while i < len(tex):
        c = tex[i]
        if c == '{':
            depth += 1
            i += 1
            continue
        if c == '}':
            depth -= 1
            if script != 1.0 and depth < group:
                script = 1.0
            i += 1
            continue
        if c in '_^':
            script, sub = _SCRIPT, True
            group = depth + 1 if i + 1 < len(tex) and tex[i + 1] == '{' else -1
            i += 1
            continue
        if c == '\\':
            j = i + 1
            while j < len(tex) and tex[j].isalpha():
                j += 1
            name = tex[i + 1:j]
            if name in ('delta', 'partial', 'phi', 'psi', 'lambda', 'beta'):
                tall = True
            if name not in ('!', ',', ';', ' '):
                w += _ATOM_MM * script
            i = max(j, i + 2)
        elif c in ' $':
            i += 1
            continue
        else:
            if c in '()[]':
                w += _PAREN_MM * script
                tall = True
            elif c.isdigit():
                w += _DIGIT_MM * script
            elif c.isalpha():
                w += _ATOM_MM * script
            else:
                w += _DIGIT_MM * script
            i += 1
        if group == -1:               # a one-atom script ends after it
            script, group = 1.0, 0
    h = _LINE_MM + (_DEEP_MM if sub else 0.0) + (_TALL_MM if tall else 0.0)
    return max(w, 1.0), h


def _label_box(vx, vy, angle_deg, dist_pt, w, h, dot_r):
    """Box a tikz ``label=`` option puts the text in, in printed mm.

    TikZ anchors the label node by its own border in the OPPOSITE direction,
    so the box centre sits one border-crossing further out than the gap: this
    reproduces that, which is what makes the diagonal angles score honestly
    against the cardinal ones (a box reaches further at 45 degrees).

    The border it anchors by is the NODE's, which is the text grown by
    ``inner sep`` -- so the border crossing is computed on the padded box
    while the box returned, the one that has to clear the drawing, stays the
    size of the ink.  Conflating the two printed every label about a
    millimetre further out than it was scored, which is how labels ended up
    across propagators that the placement believed it had avoided.
    """
    import math as _m
    a = _m.radians(angle_deg)
    ca, sa = _m.cos(a), _m.sin(a)
    hw, hh = w / 2.0, h / 2.0
    bw, bh = hw + _INNER_SEP_MM, hh + _INNER_SEP_MM      # the NODE's border
    tx = abs(bw / ca) if abs(ca) > 1e-9 else 1e18
    ty = abs(bh / sa) if abs(sa) > 1e-9 else 1e18
    reach = dot_r + dist_pt * _PT_MM + min(tx, ty)
    cx, cy = vx + reach * ca, vy + reach * sa
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _box_overlap(a, b):
    """Area two boxes share."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _seg_in_box(p, q, box):
    """Length of segment ``p``-``q`` inside ``box`` (Liang-Barsky)."""
    import math as _m
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for num, den in ((p[0] - x0, -dx), (x1 - p[0], dx),
                     (p[1] - y0, -dy), (y1 - p[1], dy)):
        if den == 0:
            if num < 0:
                return 0.0
            continue
        t = num / den
        if den < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return 0.0
    return (t1 - t0) * _m.hypot(dx, dy)


def place_labels(anchors, polylines, dots, fixed_boxes=(), rounds=4,
                 dot_r=DOT_RADIUS_MM):
    """Choose an (angle, distance) for every vertex label, jointly.

    Greedy one-at-a-time placement gives the first vertex the best spot and
    leaves the last with whatever is free, which on a dense panel is a
    position across two propagators.  This scores the same candidates but
    then SWEEPS: each label is re-placed against the boxes the others
    currently occupy, repeatedly, until nothing moves.  That is coordinate
    descent on the joint cost -- still not a global optimum, but it escapes
    the orderings a single pass cannot.

    ``anchors`` maps a key to ``(x_mm, y_mm, text_width_mm, text_height_mm)``.
    Returns ``{key: (angle_deg, distance_pt)}``.
    """
    keys = list(anchors)
    cands = [(a, d) for d in _LABEL_DISTANCES_PT for a in _LABEL_ANGLES]
    # Prefer a near, canonical placement: the tie-break that keeps a clean
    # panel looking exactly as it did before this pass existed.
    rank = {(a, d): (_LABEL_DISTANCES_PT.index(d) * len(_LABEL_ANGLES)
                     + _LABEL_ANGLES.index(a))
            for a, d in cands}

    def _is(cx, cy, owner):
        return abs(cx - owner[0]) < 1e-9 and abs(cy - owner[1]) < 1e-9

    def cost(box, others, owner):
        import math as _m
        # Judge collisions on a slightly grown box: two labels that merely
        # touch are already hard to read as two.
        grown = (box[0] - _LABEL_MARGIN_MM, box[1] - _LABEL_MARGIN_MM,
                 box[2] + _LABEL_MARGIN_MM, box[3] + _LABEL_MARGIN_MM)
        c = 0.0
        for b in others:
            c += 3.0 * _box_overlap(grown, b)        # label on a label
        for b in fixed_boxes:
            c += 3.0 * _box_overlap(grown, b)
        for (cx, cy) in dots:
            if _is(cx, cy, owner):      # a label always abuts its OWN dot
                continue
            c += 6.0 * _box_overlap(grown, (cx - dot_r, cy - dot_r,
                                            cx + dot_r, cy + dot_r))
        for poly in polylines:
            for p, q in zip(poly, poly[1:]):
                # A propagator running through a label is not a near miss,
                # it is a misread: the reader cannot tell the symbol from the
                # graph.  Charged per mm of line inside the box, heavily
                # enough to outrank the mild label-on-label crowding the
                # placer would otherwise prefer it to.
                c += 4.0 * _seg_in_box(p, q, box)
        # OWNERSHIP.  A label that ends up nearer some other vertex than its
        # own names the wrong vertex, which no amount of clearance fixes --
        # and on a dense panel two labels can sit side by side between the
        # pair they belong to, with nothing to say which is which.
        mx, my = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        own = _m.hypot(mx - owner[0], my - owner[1])
        for (cx, cy) in dots:
            if _is(cx, cy, owner):
                continue
            slack = own * _OWNERSHIP - _m.hypot(mx - cx, my - cy)
            if slack > 0.0:
                c += 1.5 * slack
        return c

    chosen = {}
    for k in keys:                                   # first pass: greedy
        x, y, w, h = anchors[k]
        boxes = [_label_box(*anchors[j][:2], *chosen[j], *anchors[j][2:],
                            dot_r) for j in chosen]
        best = None
        for a, d in cands:
            box = _label_box(x, y, a, d, w, h, dot_r)
            sc = (cost(box, boxes, (x, y)), rank[(a, d)])
            if best is None or sc < best[0]:
                best = (sc, (a, d))
        chosen[k] = best[1]

    for _ in range(rounds):                          # then sweep to a fixpoint
        moved = False
        for k in keys:
            x, y, w, h = anchors[k]
            boxes = [_label_box(*anchors[j][:2], *chosen[j], *anchors[j][2:],
                                dot_r) for j in keys if j != k]
            best = None
            for a, d in cands:
                box = _label_box(x, y, a, d, w, h, dot_r)
                sc = (cost(box, boxes, (x, y)), rank[(a, d)])
                if best is None or sc < best[0]:
                    best = (sc, (a, d))
            if best[1] != chosen[k]:
                chosen[k], moved = best[1], True
        if not moved:
            break
    return chosen


# ── the path pgf actually draws, and where the arrowhead lands ──────
# tikz-feynman marks the `fermion` arrow at t = 0.5.  The layered layout is
# symmetric, so two edges that cross between the same pair of layers cross at
# BOTH their midpoints -- which stacks their two arrowheads on one point, and
# a solid triangular blob at a line crossing reads as a VERTEX.  Measured over
# the 66 printed two-loop panels: 34 collisions on 30 of them, 13 at exactly
# 0.00 mm.  The cure is to slide the arrowheads along their own lines; an
# arrow's position carries no meaning, only its direction does.

_ARROW_CTRL = 0.3915       # pgf's `to[bend]` control point, as a chord fraction
# Tried in order, so an edge keeps the canonical midpoint unless it must move.
_ARROW_POSITIONS = (0.5, 0.42, 0.58, 0.35, 0.65, 0.28, 0.72, 0.22, 0.78)
# Clearance ON TOP of the ink, so two marks do not merely GRAZE.  A `fermion`
# head measures 1.905 mm both along and across the line (measured off an
# 800 dpi render of a one-edge picture), so a rule of `centre distance >=
# ARROWHEAD_MM` is satisfied exactly when two solid triangles touch corner to
# corner -- which is what the two heads of a bubble were doing, filling its
# lens with one black bowtie.  A panel is additionally shrunk to its column
# by `tikz_figure` (measured 0.78-0.90 over the 66-panel figure), so the gap
# has to be worth having before that shrink eats a fifth of it.
_ARROW_GAP_MM = 0.6


# The bow of a parallel pair is defined ONCE, in the layout module: it is
# geometry the layout must reason about (a bubble's arc sweeps a lens that
# has to stay clear of other vertices) and geometry this module strokes, and
# two copies of it would drift.  ``_BOW``, ``_MAX_BEND`` and
# ``MIN_BUBBLE_MM`` are imported above and re-exported here unchanged.


def edge_bends(edges, bend_angle, pos=None, mm=None,
               min_lens_mm=MIN_BUBBLE_MM):
    """Signed bend in degrees per drawn edge (positive = tikz ``bend left``).

    Parallel edges between one pair of vertices are bowed symmetrically apart
    so the pair reads as a bubble instead of a single line.  Returned as data
    so the arrow placement can reconstruct the curve the reader sees rather
    than the straight chord.

    A FIXED angle bows by a fixed FRACTION of the edge, so a short bubble
    gets a proportionally narrow lens: at 14 degrees the tightest bubble in
    the two-loop figure opened to 1.33 mm, less than the width of the vertex
    dots at either end, and read as a single thick line.  Given ``pos`` and a
    millimetre scale, the angle is therefore raised on short pairs until the
    printed gap reaches ``min_lens_mm``, capped so a bubble never balloons.
    """
    import math as _m
    from collections import Counter
    multiplicity, drawn, out = Counter(edges), Counter(), []
    angle_of = {}
    for (u, v), n in multiplicity.items():
        a = bend_angle
        if n > 1 and pos is not None and mm and u in pos and v in pos:
            L = _m.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]) * mm
            if L > 0:
                a = bubble_bend_deg(L, bend_angle, min_lens_mm)
        angle_of[(u, v)] = a
    for (u, v) in edges:
        n, j = multiplicity[(u, v)], drawn[(u, v)]
        drawn[(u, v)] += 1
        b = 0
        if n > 1:
            offset = j - (n - 1) / 2.0
            if abs(offset) > 1e-9:
                mag = int(round(angle_of[(u, v)] * abs(offset) * 2))
                b = mag if offset < 0 else -mag
        out.append(b)
    return out


def path_point(pu, pv, bend_deg, t):
    """Point at parameter ``t`` on the path pgf draws from ``pu`` to ``pv``.

    A straight ``--`` is a line; a ``to[bend=x]`` is the cubic Bezier pgf
    builds with its control points at ``_ARROW_CTRL`` of the chord, turned by
    the bend angle.  Reconstructing it exactly is what makes the arrow
    positions below correct on a bowed bubble as well as a straight edge.
    """
    import math as _m
    if not bend_deg:
        return (pu[0] + t * (pv[0] - pu[0]), pu[1] + t * (pv[1] - pu[1]))
    L = _m.hypot(pv[0] - pu[0], pv[1] - pu[1])
    th = _m.atan2(pv[1] - pu[1], pv[0] - pu[0])
    a = _m.radians(bend_deg)
    c1 = (pu[0] + _ARROW_CTRL * L * _m.cos(th + a),
          pu[1] + _ARROW_CTRL * L * _m.sin(th + a))
    c2 = (pv[0] - _ARROW_CTRL * L * _m.cos(th - a),
          pv[1] - _ARROW_CTRL * L * _m.sin(th - a))
    m = 1.0 - t
    w = (m * m * m, 3 * m * m * t, 3 * m * t * t, t * t * t)
    return (w[0] * pu[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * pv[0],
            w[0] * pu[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * pv[1])


def arrow_positions(pos, edges, bends, mm=None):
    """Where along each edge to mark its arrow, in path parameter ``t``.

    Greedy, and deliberately biased: every edge is offered the midpoint
    first, so a diagram with no collision is drawn exactly as before.  An
    edge moves only when its arrowhead would land within an arrowhead's width
    of one already placed, on a vertex dot it is not entering, or ON ANOTHER
    PROPAGATOR -- all three of which invent structure that is not in the
    graph.  The last is the commonest: a symmetric layered drawing crosses
    two edges at their mutual midpoints, so separating the two ARROWHEADS
    from each other still leaves one of them centred on the crossing, and a
    solid triangle sitting on a line crossing reads as a VERTEX.  The line
    an arrow belongs to, and any line sharing an endpoint with it, are not
    obstacles: those converge on a shared dot by construction, and charging
    for them would drive every arrow onto the far ends of its propagator.
    """
    import math as _m
    if mm is None:
        mm = mm_per_unit(pos)
    dots = [p for p in pos.values()]
    # The drawn path of every edge, sampled -- an obstacle for the arrowheads
    # of the edges it does not touch.
    paths = [[path_point(pos[a], pos[c], bd, k / 24.0) for k in range(25)]
             if a in pos and c in pos else None
             for (a, c), bd in zip(edges, bends)]
    placed, out = [], []
    for i, ((u, v), b) in enumerate(zip(edges, bends)):
        if u not in pos or v not in pos:
            out.append(0.5)
            continue
        foreign = [P for j, P in enumerate(paths)
                   if P is not None and j != i
                   and u not in edges[j] and v not in edges[j]]
        best_t, best_score = _ARROW_POSITIONS[0], None
        for t in _ARROW_POSITIONS:
            q = path_point(pos[u], pos[v], b, t)
            score = min(
                [_m.hypot(q[0] - r[0], q[1] - r[1]) * mm
                 - (ARROWHEAD_MM + _ARROW_GAP_MM)
                 for r in placed]
                + [_m.hypot(q[0] - c[0], q[1] - c[1]) * mm
                   - (DOT_RADIUS_MM + ARROWHEAD_MM / 2.0 + _ARROW_GAP_MM)
                   for c in dots]
                + [min(_seg_distance(q, P[k], P[k + 1])
                       for k in range(len(P) - 1)) * mm
                   - (ARROWHEAD_MM / 2.0 + _ARROW_GAP_MM)
                   for P in foreign]
                or [9e9])
            if best_score is None or score > best_score:
                best_t, best_score = t, score
            if score >= 0.0:                  # good enough; keep it central
                break
        out.append(best_t)
        placed.append(path_point(pos[u], pos[v], b, best_t))
    return out


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


def edge_style_map(diagrams):
    """Assign a distinct line style to each field pairing present.

    In a multi-field model different edges are different components of G, and
    a label alone makes the reader parse subscripts to see the structure.
    Giving each pairing its own stroke makes it visible at a glance.

    Accepts one diagram or MANY, and the difference matters: built per panel,
    the styles are handed out over the pairings THAT panel happens to
    contain, so a solid line means G_xx in one panel and G_xy in the next
    while the key claims one meaning for the whole figure.  Measured on the
    24-panel two-field figure, `solid` covered three different components.
    Pass the whole set -- the same rule as ``vertex_symbol_map``.
    """
    if not isinstance(diagrams, (list, tuple)):
        diagrams = [diagrams]
    pairs = set()
    for dia in diagrams:
        et = getattr(dia, 'edge_types', None) or {}
        pairs.update((_field_symbol(a), _field_symbol(b))
                     for a, b in et.values())
    return {p: _EDGE_STYLES[i % len(_EDGE_STYLES)]
            for i, p in enumerate(sorted(pairs))}


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
                    edge_styles=None, indent='  '):
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
    edge_styles : dict or None
        A line-style map shared across a whole figure, for the same reason:
        built per panel, a stroke names whichever component that panel
        happened to contain.  ``tikz_figure`` supplies it automatically.
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
    style_of = ((edge_styles if edge_styles is not None
                 else edge_style_map(diagram))
                if style_by_field else {})
    # Each distinct propagator symbol is named once.  Repeating one symbol on
    # every edge tells the reader nothing after the first time, and once the
    # line STYLE also encodes the component the labels are pure clutter.
    _seen_edge_labels = set()
    ext_legs = getattr(diagram, 'external_legs', None) or {}

    # ── what every label SAYS ───────────────────────────────────────
    # Decided before anything is emitted, because the placement below is
    # joint: it needs every box at once, and a label's size depends on its
    # text.
    ext_text = {}
    for idx, v in enumerate(sorted(leaves), start=1):
        if external_label == 'auto':
            sym = _field_symbol(ext_legs.get(v)) or r'\phi'
            ext_text[v] = r'\delta %s(y_{%d})' % (sym, idx)
        else:
            ext_text[v] = external_label % idx

    # ``show_factors='auto'``: print each DISTINCT factor once.  Every
    # interaction vertex of a one-species model carries the same coefficient,
    # so labelling all of them repeats one value as many times as the loop
    # order allows and, past one loop, the copies collide with each other.
    # One instance of each distinct value loses no information.
    _seen_factors = set()
    vsym = (vertex_symbols if vertex_symbols is not None
            else (vertex_symbol_map(diagram, symbol_map, keyed=True)
                  if symbolic_factors else {}))
    int_text = {}
    for v in sorted(set(D.vertices()) - leaf_set):
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
        int_text[v] = label

    # ── where every label GOES ──────────────────────────────────────
    # In printed millimetres: node text is not affected by ``scale=``, so in
    # a shrunk panel the labels keep their size while the diagram loses its,
    # and only a millimetre model gets the collisions right.
    mm = mm_per_unit(pos)
    _P = {u: (x * mm, y * mm) for u, (x, y) in pos.items()}
    _bends = edge_bends(edge_list, bend_angle, pos, mm)
    _polys = [[tuple(c * mm for c in path_point(pos[a], pos[b], bd, k / 12.0))
               for k in range(13)]
              for (a, b), bd in zip(edge_list, _bends)]
    _ext_pt = _distance_pt(external_label_distance)

    def _pad(t):
        w, h = _label_extent_mm(t)
        return w * 1.08 + 0.5, h * 1.08 + 0.3     # the estimator runs ~5% low

    # Arrowheads are obstacles in their own right.  A propagator is a
    # zero-width polyline to `_seg_in_box`, but its arrow is a solid triangle
    # 1.91 mm across, so a label scored clear of the LINE could still be
    # printed through the head -- seen on panel 17 of the two-loop figure.
    _arrows = arrow_positions(pos, edge_list, _bends, mm)
    _ahw = ARROWHEAD_MM / 2.0
    arrow_boxes = []
    for (a, b), bd, t in zip(edge_list, _bends, _arrows):
        if a not in pos or b not in pos:
            continue
        q = path_point(pos[a], pos[b], bd, t)
        qx, qy = q[0] * mm, q[1] * mm
        arrow_boxes.append((qx - _ahw, qy - _ahw, qx + _ahw, qy + _ahw))

    ext_boxes = [_label_box(_P[v][0], _P[v][1], 180, _ext_pt,
                            *_pad(t), DOT_RADIUS_MM)
                 for v, t in ext_text.items() if t]
    anchors = {v: (_P[v][0], _P[v][1]) + _pad(t)
               for v, t in int_text.items() if t}
    if factor_label_angle == 'auto':
        placed = place_labels(anchors, _polys, list(_P.values()),
                              fixed_boxes=ext_boxes + arrow_boxes)
    else:
        placed = {v: (factor_label_angle, 1.0) for v in anchors}

    # ── vertices ────────────────────────────────────────────────────
    for v in sorted(leaves):
        x, y = pos[v]
        # The label rides on a ``label=`` option, NOT in the node body: an
        # ``[empty dot]`` sized to contain ``\delta x(y_1)`` becomes a huge
        # circle with text inside it.  Placed at 180 degrees it sits to the
        # LEFT of a small circle, outside the diagram, where nothing crosses it.
        lines.append(
            indent * 2 + r'\vertex [empty dot, label={[label distance=%s]'
                         r'180:\(%s\)}] (%s) at (%.3f, %.3f) {};'
            % (external_label_distance, ext_text[v], _node_name(v), x, y))

    for v in sorted(set(D.vertices()) - leaf_set):
        x, y = pos[v]
        label = int_text[v]
        # A ``[dot]`` vertex is a FILLED node: anything in its node body is
        # drawn inside the ink and invisible.  The factor must therefore ride
        # on a ``label=`` option, placed clear of the propagator lines
        # converging on it.
        opts = 'dot'
        if label:
            angle, dist = placed[v]
            opts += r', label={[label distance=%gpt]%d:\(%s\)}' % (
                dist, angle, label)
        lines.append(indent * 2 + r'\vertex [%s] (%s) at (%.3f, %.3f) {};'
                     % (opts, _node_name(v), x, y))

    # ── edges ───────────────────────────────────────────────────────
    # Edge direction is the retarded propagator's time arrow (earlier ->
    # later).  Since later times sit further LEFT, drawing tail -> head
    # already points the arrow leftward; no reversal is needed.

    lines.append(indent * 2 + r'\diagram* {')
    edges = edge_list
    # Parallel edges (a loop between two vertices) must be bowed apart or
    # tikz-feynman draws them on top of each other.  Bend symmetrically about
    # the straight line so the pair reads as a bubble.
    from collections import Counter
    drawn = Counter()
    # Bow parallel propagators apart symmetrically.  NOT ``half left/right``:
    # that is a SEMICIRCULAR arc, so between two distant vertices it balloons
    # into a circle bigger than the diagram.  A fixed modest angle bows by the
    # same visual amount at any separation.
    bends = _bends
    arrows = _arrows
    for i, (u, v) in enumerate(edges):
        j = drawn[(u, v)]
        drawn[(u, v)] += 1
        b = bends[i]
        bend = (', bend %s=%d' % ('left' if b > 0 else 'right', abs(b))
                if b else '')
        # Per-edge label: with ``propagator_label='auto'`` different lines
        # are different matrix elements of G and must be named separately.
        lab = _edge_label(diagram, (u, v, j), propagator_label)
        if lab and propagator_label == 'auto':
            if lab in _seen_edge_labels:
                lab = None
            else:
                _seen_edge_labels.add(lab)
        # ``fermion`` IS ``plain`` plus an arrow at t = 0.5, and the two keys
        # do not compose -- asking for both marks the line TWICE.  An edge
        # whose arrow has been slid away from the midpoint therefore has to
        # be spelled out the long way.
        t = arrows[i]
        opts = (['fermion'] if abs(t - 0.5) < 1e-9
                else ['plain', 'with arrow=%.3f' % t])
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

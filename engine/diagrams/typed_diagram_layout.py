"""Geometry for drawing a typed diagram.

Pure layout: this module assigns an ``(x, y)`` to every vertex and never emits
markup, so the placement rules can be tested on coordinates alone.

Convention (matches the figures in the paper)
---------------------------------------------
**Time flows right to left.**  A prediagram's edges are oriented along the
retarded propagator, earlier -> later, so a vertex's *causal depth* (the
longest directed path reaching it from any source) increases with time.  We
map larger depth to smaller ``x``: sources rightmost, external legs flush
left on a shared ``x`` so panels in a figure array line up.

Within that, this is a layered graph drawing in the Sugiyama sense, and the
two passes after layering matter as much as the layering itself:

1. **Ordering** -- vertices inside a layer are permuted to reduce edge
   crossings, by iterated barycentre sweeps.  Ordering by vertex id instead
   (the obvious thing, and what this module did first) leaves crossings that
   carry no physical meaning and read as mistakes.
2. **Coordinates** -- y is then relaxed toward each vertex's neighbours,
   subject to a minimum separation and to the ordering fixed in step 1.  A
   rigid symmetric spread pins every source onto the same few rows and bends
   the edges reaching them; letting them slide straightens the picture.
"""

import itertools
import math
from collections import defaultdict

__all__ = ['causal_depths', 'layout_typed_diagram', 'layout_prediagram',
           'panel_scale', 'mm_per_unit', 'legibility_penalty',
           'vertex_edge_clearances', 'fan_separations',
           'DX', 'DY', 'DY_EXTERNAL', 'ASPECT',
           'DOT_RADIUS_MM', 'ARROWHEAD_MM', 'PANEL_BUDGET_COLS',
           'PANEL_MAX_SCALE']

# Column pitch (x) and row pitch (y) in TikZ units.
DX = 2.3
DY = 1.25
# External vertices sit BESIDE their label rather than inside it, so the node
# is a small circle and needs only slightly more room than an internal dot.
DY_EXTERNAL = 1.5
# Minimum height/width for a finished diagram.  Without this a deep, weakly
# branching diagram is drawn many columns wide and two rows tall, and shrinks
# into an illegible sliver inside a panel.
ASPECT = 0.55

# ── printed size of the ink ──────────────────────────────────────────
# A layout is scale-free, but whether two lines are DISTINGUISHABLE is not:
# it depends on how big the marks are once printed.  Both constants below
# were measured off a 400 dpi render of a one-edge picture at scale 1, so
# they are the real tikz-feynman ink, not a guess.
DOT_RADIUS_MM = 0.825          # radius of a filled [dot] vertex
ARROWHEAD_MM = 1.91            # perpendicular width of a `fermion` arrowhead

# How a panel of an array is printed.  ``tikz_figure._auto_scale`` shrinks a
# drawing wider than ``PANEL_BUDGET_COLS`` columns to fit its column, and
# otherwise draws it at ``PANEL_MAX_SCALE``; the two live here so the layout
# can reason in millimetres and so the pair cannot drift apart.
PANEL_BUDGET_COLS = 2.0
PANEL_MAX_SCALE = 0.75


def panel_scale(pos, budget_cols=PANEL_BUDGET_COLS,
                max_scale=PANEL_MAX_SCALE):
    """``tikzpicture`` scale a panel holding this drawing is printed at."""
    if not pos:
        return max_scale
    xs = [x for x, _ in pos.values()]
    span = (max(xs) - min(xs)) / DX
    if span <= budget_cols:
        return max_scale
    return budget_cols / span


def mm_per_unit(pos):
    """Printed millimetres per layout unit for a drawing in a panel array.

    A TikZ unit is 1 cm before scaling, so this is just ``10 * scale`` -- but
    routing it through :func:`panel_scale` means a wide diagram, which is
    shrunk to fit, is correctly judged to have SMALLER printed features than
    a narrow one at the same layout separation.
    """
    return 10.0 * panel_scale(pos)


def _unpack(diagram):
    """Return ``(D, leaves)`` from a TypedDiagram or a raw prediagram tuple."""
    prediagram = getattr(diagram, 'prediagram', diagram)
    D, _G, leaves, _internal = prediagram
    return D, list(leaves)


def causal_depths(D):
    """Longest-path depth of every vertex from the set of sources.

    Sources are the in-degree-0 vertices.  The orientation constraints make
    the digraph acyclic, so relaxing over a topological order is exact.
    """
    depth = {v: 0 for v in D.vertices()}
    try:
        order = D.topological_sort()
    except Exception:
        order = list(D.vertices())
    for u in order:
        for v in D.neighbors_out(u):
            if depth[u] + 1 > depth[v]:
                depth[v] = depth[u] + 1
    return depth


def _neighbours(D):
    """Undirected adjacency, one entry per edge copy."""
    adj = defaultdict(list)
    for u, v in D.edges(labels=False):
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _order_layers(layer_of, layers, adj, sweeps=6):
    """Permute each layer to reduce crossings (barycentre heuristic).

    Sweep down then up repeatedly, placing each vertex at the mean index of
    its neighbours in the adjacent layer.  Cheap, and at these diagram sizes
    it removes essentially all avoidable crossings.
    """
    index = {}
    for d in layers:
        for i, v in enumerate(layers[d]):
            index[v] = i

    depths = sorted(layers)
    for sweep in range(sweeps):
        seq = depths if sweep % 2 == 0 else list(reversed(depths))
        for d in seq:
            ref = d - 1 if sweep % 2 == 0 else d + 1
            if ref not in layers:
                continue

            def bary(v, _ref=ref):
                ns = [index[n] for n in adj[v] if layer_of.get(n) == _ref]
                return sum(ns) / len(ns) if ns else float(index[v])

            layers[d].sort(key=bary)
            for i, v in enumerate(layers[d]):
                index[v] = i
    return index


def _segments_cross(p, q, r, t):
    """Do open segments p-q and r-t properly cross?  Shared endpoints do not."""
    if p in (r, t) or q in (r, t):
        return False

    def side(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    d1, d2 = side(p, q, r), side(p, q, t)
    d3, d4 = side(r, t, p), side(r, t, q)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _geometric_crossings(pos, edges):
    """Count crossings among the straight segments actually drawn.

    Layer-by-layer counting -- even with dummy nodes -- optimises a ROUTE
    that the renderer then ignores, because a multi-layer edge is drawn as
    one straight line from its real endpoints.  Scoring the drawn geometry
    instead optimises what the reader sees, and needs no dummies at all.
    """
    segs = [(pos[u], pos[v]) for u, v in edges if u in pos and v in pos]
    n = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if _segments_cross(segs[i][0], segs[i][1],
                               segs[j][0], segs[j][1]):
                n += 1
    return n


def _seg_distance(p, a, b):
    """Distance from point ``p`` to the segment ``a``-``b``."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    ux, uy = bx - ax, by - ay
    den = ux * ux + uy * uy
    if den == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * ux + (py - ay) * uy) / den))
    return math.hypot(px - ax - t * ux, py - ay - t * uy)


def vertex_edge_clearances(pos, edges, mm=None):
    """Printed distance from every vertex to every edge it is NOT on.

    Yields ``(mm, vertex, edge)``.  A value below the vertex dot's own radius
    means the propagator is drawn THROUGH the dot's ink, and the reader sees
    one line passing behind a vertex as two lines meeting at it -- a
    different graph.
    """
    if mm is None:
        mm = mm_per_unit(pos)
    for (u, v) in sorted({tuple(e) for e in edges}):
        if u not in pos or v not in pos:
            continue
        for w, q in pos.items():
            if w == u or w == v:
                continue
            yield _seg_distance(q, pos[u], pos[v]) * mm, w, (u, v)


def fan_separations(pos, edges, mm=None):
    """Printed gap between two edges leaving a shared vertex.

    Measured at the nearer of the two far endpoints -- ``min(L1, L2) *
    sin(theta)`` -- because that is where the pair is still drawn on top of
    each other.  Below one arrowhead width the two propagators print as a
    single line with a sliver, and the diagram reads as having one propagator
    fewer than it has.  Yields ``(mm, edge, edge)``.
    """
    if mm is None:
        mm = mm_per_unit(pos)
    es = sorted({tuple(e) for e in edges})
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            a, b = es[i], es[j]
            shared = set(a) & set(b)
            if not shared:
                continue
            s = shared.pop()
            fa = a[1] if a[0] == s else a[0]
            fb = b[1] if b[0] == s else b[0]
            if fa == fb or s not in pos or fa not in pos or fb not in pos:
                continue
            sx, sy = pos[s]
            ax, ay = pos[fa][0] - sx, pos[fa][1] - sy
            bx, by = pos[fb][0] - sx, pos[fb][1] - sy
            la, lb = math.hypot(ax, ay), math.hypot(bx, by)
            if la == 0.0 or lb == 0.0:
                continue
            sin_t = abs(ax * by - ay * bx) / (la * lb)
            yield min(la, lb) * sin_t * mm, a, b


def legibility_penalty(pos, edges, mm=None):
    """How badly this drawing misrepresents its own graph, in mm^2.

    ``_geometric_crossings`` is blind to the failure that actually ruins a
    panel: a configuration where a vertex lands ON an unrelated propagator,
    or two edges leave a vertex almost parallel, scores a PERFECT zero
    crossings while printing as a different graph entirely.  This is the
    missing term.  Both contributions are squared hinges in printed
    millimetres, so they are commensurable and vanish exactly when every
    feature is separated by more than the ink that would merge it.
    """
    if mm is None:
        mm = mm_per_unit(pos)
    pen = 0.0
    for d, _w, _e in vertex_edge_clearances(pos, edges, mm):
        gap = 2.0 * DOT_RADIUS_MM - d          # clear the whole dot, not half
        if gap > 0.0:
            pen += gap * gap
    for d, _a, _b in fan_separations(pos, edges, mm):
        gap = ARROWHEAD_MM - d
        if gap > 0.0:
            pen += gap * gap
    return pen


def _nudge_y(layers, layer_of, y_of, D_edges, dx, pitch_of, rounds=4):
    """Slide vertices off the lines they are sitting on.

    The repair for the case no ORDERING can fix: when the only y consistent
    with the chosen column order puts a vertex on a propagator, a small
    perpendicular offset removes the ambiguity, and because it never reorders
    a column the minimum-crossing ordering still holds.  Guarded twice
    anyway -- a move is kept only if the DRAWN crossing count does not rise
    and the penalty strictly falls -- so this pass can only improve a
    drawing.
    """
    def pos_of():
        return {v: (-layer_of[v] * dx, y_of[v]) for v in y_of}

    pos = pos_of()
    base_cross = _geometric_crossings(pos, D_edges)
    best = legibility_penalty(pos, D_edges)
    if best <= 0.0:
        return y_of
    fracs = (0.18, -0.18, 0.36, -0.36, 0.55, -0.55, 0.8, -0.8)
    for _ in range(rounds):
        moved = False
        for d in sorted(layers):
            col = layers[d]
            pitch = pitch_of(d)
            for i, v in enumerate(col):
                if v not in y_of:
                    continue
                lo = (y_of[col[i - 1]] + pitch) if i > 0 else -1e18
                hi = (y_of[col[i + 1]] - pitch) if i < len(col) - 1 else 1e18
                y0 = y_of[v]
                pick, pick_pen = y0, best
                for f in fracs:
                    yt = y0 + f * pitch
                    if yt < lo - 1e-9 or yt > hi + 1e-9:
                        continue
                    y_of[v] = yt
                    cand = pos_of()
                    if _geometric_crossings(cand, D_edges) > base_cross:
                        continue
                    pen = legibility_penalty(cand, D_edges)
                    if pen < pick_pen - 1e-9:
                        pick, pick_pen = yt, pen
                y_of[v] = pick
                if pick != y0:
                    best, moved = pick_pen, True
        if not moved or best <= 0.0:
            break
    return y_of


def _with_dummies(layer_of, D_edges):
    """Split multi-layer edges with dummy nodes, one per intermediate layer.

    Without this an edge spanning several layers -- a source wired straight
    to an external, say -- is invisible to both the barycentre pass and the
    crossing count, because those only ever compare ADJACENT layers.  Such an
    edge sails over the intermediate vertices and crosses whatever is in the
    way, which is exactly the crossing that survives an otherwise clean
    ordering.  Sugiyama's answer is to make the long edge into a chain of
    unit-length segments so the intermediate layers can order around it.

    Returns ``(layer_of2, segments)`` where segments are all unit-length.
    """
    layer_of2 = dict(layer_of)
    segments = []
    for n, (u, v) in enumerate(D_edges):
        lu, lv = layer_of[u], layer_of[v]
        lo, hi = (lu, lv) if lu <= lv else (lv, lu)
        a, b = (u, v) if lu <= lv else (v, u)
        if hi - lo <= 1:
            segments.append((a, b))
            continue
        prev = a
        for d in range(lo + 1, hi):
            dummy = ('_d', n, d)
            layer_of2[dummy] = d
            segments.append((prev, dummy))
            prev = dummy
        segments.append((prev, b))
    return layer_of2, segments


def _layer_pair_crossings(upper, lower, layer_of, adj_pairs):
    """Crossings between two adjacent layers, given their orderings."""
    iu = {v: i for i, v in enumerate(upper)}
    il = {v: i for i, v in enumerate(lower)}
    es = [(iu[a], il[b]) for a, b in adj_pairs
          if a in iu and b in il]
    n = 0
    for i in range(len(es)):
        a1, b1 = es[i]
        for j in range(i + 1, len(es)):
            a2, b2 = es[j]
            if (a1 - a2) * (b1 - b2) < 0:
                n += 1
    return n


def _edge_pairs_between(D_edges, layer_of, d_up, d_lo):
    """Edges spanning layers ``d_up`` -> ``d_lo``, as (upper, lower) pairs."""
    out = []
    for u, v in D_edges:
        lu, lv = layer_of.get(u), layer_of.get(v)
        if lu == d_up and lv == d_lo:
            out.append((u, v))
        elif lv == d_up and lu == d_lo:
            out.append((v, u))
    return out


def _total_crossings(layers, layer_of, D_edges):
    depths = sorted(layers)
    tot = 0
    for a, b in zip(depths, depths[1:]):
        pairs = _edge_pairs_between(D_edges, layer_of, a, b)
        tot += _layer_pair_crossings(layers[a], layers[b], layer_of, pairs)
    return tot


def _exhaustive_order(layers, layer_of, D_edges, budget=50000):
    """Exact minimum-crossing ordering, when the search space is small enough.

    Diagram layers here are tiny -- typically one or two vertices -- so the
    whole ordering space is usually a few dozen permutations.  Hill-climbing
    that is both pointless and unreliable: measured on the two-point one-loop
    set, adjacent-swap refinement stalled at 2 crossings on a diagram whose
    true minimum is 0, needing only the external pair exchanged, because no
    SINGLE swap improved the count.  Enumerate instead whenever the product
    of layer factorials fits ``budget``, and fall back to the heuristic when
    it does not.

    Returns the achieved crossing count, or ``None`` if the space was too big.
    """
    import itertools, math
    depths = sorted(layers)
    total = 1
    for d in depths:
        total *= math.factorial(len(layers[d]))
        if total > budget:
            return None
    best_n, best = None, None
    for combo in itertools.product(*(itertools.permutations(layers[d])
                                     for d in depths)):
        cand = {d: list(c) for d, c in zip(depths, combo)}
        n = _total_crossings(cand, layer_of, D_edges)
        if best_n is None or n < best_n:
            best_n, best = n, cand
            if n == 0:
                break
    for d in depths:
        layers[d][:] = best[d]
    return best_n


def _refine_by_swaps(layers, layer_of, D_edges, max_rounds=12):
    """Adjacent-transposition refinement against a real crossing count.

    The barycentre pass is only a heuristic -- it has no crossing objective
    and readily stops at an ordering with an obvious avoidable crossing (for
    the two-point one-loop set, one panel needed nothing more than the two
    SOURCES exchanged).  Here we try every adjacent swap in every layer and
    keep the ones that actually reduce the count, so such cases are removed
    automatically rather than by eye.  A swap is kept only on strict
    improvement, so this can never make a drawing worse.
    """
    best = _total_crossings(layers, layer_of, D_edges)
    for _ in range(max_rounds):
        improved = False
        for d in sorted(layers):
            col = layers[d]
            for i in range(len(col) - 1):
                col[i], col[i + 1] = col[i + 1], col[i]
                cand = _total_crossings(layers, layer_of, D_edges)
                if cand < best:
                    best = cand
                    improved = True
                else:
                    col[i], col[i + 1] = col[i + 1], col[i]
        if not improved:
            break
    return best


def _relax_y(layers, adj, y_of, pitch_of, rounds=6, weight=0.3):
    """Slide vertices toward their neighbours, keeping order and separation.

    Straightens the long edges a rigid symmetric spread would bend, which is
    most of what separates a layered drawing that looks composed from one
    that looks emitted.

    ``weight`` is deliberately gentle.  Pulling hard toward the neighbour
    mean collapses every column onto the midline -- the picture gets
    straighter edges but loses the vertical separation that makes the
    vertices legible, which is the opposite of the intent.
    """
    for _ in range(rounds):
        for d in sorted(layers):
            col = layers[d]
            pitch = pitch_of(d)
            for v in col:
                ns = [y_of[n] for n in adj[v] if n in y_of]
                if ns:
                    y_of[v] = ((1.0 - weight) * y_of[v]
                               + weight * (sum(ns) / len(ns)))
            # Re-impose the ordering fixed by the barycentre pass, then the
            # minimum separation.  Order is never changed here.
            ys = sorted(y_of[v] for v in col)
            for v, y in zip(col, ys):
                y_of[v] = y
            for i in range(1, len(col)):
                lo = y_of[col[i - 1]] + pitch
                if y_of[col[i]] < lo:
                    y_of[col[i]] = lo
    return y_of


def layout_typed_diagram(diagram, dx=DX, dy=DY, order_externals=True,
                         aspect=ASPECT):
    """Assign ``(x, y)`` to every vertex of a typed diagram or prediagram.

    Parameters
    ----------
    diagram : TypedDiagram or (D, G, leaves, internal)
    dx, dy : float
        Column and row pitch.
    order_externals : bool
        Let the external column be permuted for crossing reduction.  ``False``
        pins them in leaf order, keeping panels of a figure array mutually
        consistent at the cost of extra crossings.
    aspect : float
        Minimum height/width of the finished drawing; y is stretched to reach
        it.  ``None`` disables.

    Returns
    -------
    dict
        ``{vertex: (x, y)}``.  Externals share the smallest ``x``; sources
        sit at ``x = 0``.
    """
    D, leaves = _unpack(diagram)
    leaf_set = set(leaves)
    depth = causal_depths(D)
    adj = _neighbours(D)

    internal = [v for v in D.vertices() if v not in leaf_set]
    max_internal_depth = max((depth[v] for v in internal), default=0)

    # Externals get their own layer, one column left of the deepest internal
    # vertex, so they are flush and aligned across panels.
    ext_depth = max_internal_depth + 1
    layer_of = {v: (ext_depth if v in leaf_set else depth[v])
                for v in D.vertices()}
    layers = defaultdict(list)
    for v in sorted(D.vertices()):
        layers[layer_of[v]].append(v)

    fixed_ext = sorted(leaves)
    D_edges = list(D.edges(labels=False))
    dy_ext = dy * (DY_EXTERNAL / DY)

    def pitch_of(d):
        return dy_ext if d == ext_depth else dy

    def place(cand):
        """y for one candidate ordering, after relaxation."""
        y = {}
        for d, col in cand.items():
            pitch, n = pitch_of(d), len(col)
            for i, v in enumerate(col):
                y[v] = (i - (n - 1) / 2.0) * pitch
        _relax_y(cand, adj, y, pitch_of)
        return y

    # Choose the layer orderings by the crossings ACTUALLY DRAWN.  Layers here
    # hold one or two vertices, so the whole space is a few dozen orderings and
    # an exact search is both affordable and reliable -- barycentre and
    # adjacent-swap refinement each stalled at a local minimum on this set,
    # leaving a crossing that a single exchange of the external pair removed.
    _order_layers(layer_of, layers, adj)          # good starting point
    depths = sorted(layers)
    space = 1
    for d in depths:
        space *= math.factorial(len(layers[d]))
    # Crossings first, then LEGIBILITY.  Ties on crossing count are the rule
    # rather than the exception here, and among equally-crossing orderings
    # some put a vertex straight onto a propagator; ranking the ties by
    # `legibility_penalty` picks a readable one.  Lexicographic, so the
    # proven-minimum crossing count is never traded away for tidiness.
    best = None
    if space <= 20000:
        for combo in itertools.product(*(itertools.permutations(layers[d])
                                         for d in depths)):
            cand = {d: list(c) for d, c in zip(depths, combo)}
            if not order_externals and cand[ext_depth] != fixed_ext:
                continue
            y = place(cand)
            pos_try = {v: (-layer_of[v] * dx, y[v]) for v in y}
            n = _geometric_crossings(pos_try, D_edges)
            if best is not None and n > best[0]:
                continue
            # Score the penalty in the frame the panel is actually drawn in:
            # `_fit_aspect` stretches y only, so it changes every angle.
            pen = legibility_penalty(_fit_aspect(pos_try, aspect), D_edges)
            if best is None or (n, pen) < (best[0], best[1]):
                best = (n, pen, cand, y)
                if n == 0 and pen == 0.0:
                    break
    if best is None:                              # too large: keep heuristic
        if not order_externals:
            layers[ext_depth] = fixed_ext
        y_of = place({d: list(layers[d]) for d in depths})
    else:
        for d in depths:
            layers[d][:] = best[2][d]
        y_of = best[3]

    # Stretch to the target aspect BEFORE the repair pass, so the repair sees
    # the geometry the reader will.
    k = _aspect_scale({v: (-layer_of[v] * dx, y_of[v]) for v in y_of}, aspect)
    if k != 1.0:
        for v in y_of:
            y_of[v] *= k
    _nudge_y(layers, layer_of, y_of, D_edges, dx,
             lambda d, _k=k: pitch_of(d) * _k)

    if y_of:                                   # re-centre about y = 0
        mid = 0.5 * (max(y_of.values()) + min(y_of.values()))
        for v in y_of:
            y_of[v] -= mid

    return {v: (-layer_of[v] * dx, y_of[v]) for v in D.vertices()}


def _fit_aspect(pos, target):
    """Scale y so the drawing is not a flat ribbon.

    A diagram deep in causal depth but shallow in branching comes out many
    columns wide and one or two rows tall.  Shrunk into a panel it becomes an
    unreadable sliver, and the vertex labels have nowhere to go.  Stretching y
    to a target height/width ratio fixes both, and is safe: a uniform positive
    scale on y preserves the vertex ORDER in every column, so the
    minimum-crossing ordering chosen earlier still holds exactly.
    """
    k = _aspect_scale(pos, target)
    if k == 1.0:
        return pos
    return {v: (x, y * k) for v, (x, y) in pos.items()}


def _aspect_scale(pos, target):
    """The y stretch :func:`_fit_aspect` would apply (1.0 for none)."""
    if not pos or not target:
        return 1.0
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return 1.0
    want = target * w
    return 1.0 if h >= want else want / h


layout_prediagram = layout_typed_diagram

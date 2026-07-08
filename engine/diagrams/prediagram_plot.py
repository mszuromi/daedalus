"""Contributing-prediagram plots, grouped by topology family.

For a model at correlator order ``k`` and loop order ``max_ell``, draw the
MSR-JD *prediagrams* (directed topologies) that survive the vertex/source
filter -- i.e. the ones the model can actually realise -- in the Buice/Ocker
convention: time flows RIGHT -> LEFT, with sources on the right, interaction
(internal) vertices in the middle, and external (response) legs on the left.

Roles are structural (``filter.classify_prediagram_vertices``), with a clean,
role-distinct symbolic taxonomy:
  * source      = non-leaf, in-degree 0   (filled, labelled i, ii, ...  -- right)
  * interaction = non-leaf, in-degree > 0 (filled, labelled a, b, c, ... -- middle)
  * external    = leaf                     (hollow, labelled 1, 2, ...   -- left)
Propagators carry NO symbol of their own: each is named by its endpoints, e.g.
``a->b`` (a mid-edge arrowhead gives the direction, so the two halves of a
bubble are just ``a->b`` and ``b->a``).  Diagrams are grouped by a topology
signature.

Labels are GENERIC (role + index only) -- the specific typing of any chosen
diagram (propagator a->b -> Delta_xy, source i -> K^(2), a -> phi'', external
leg 1 -> field) is a separate label-mapping table.

Layout uses graphviz 'dot' (proper layered layout) when available, falling back
to a hand-rolled SCC-aware layered layout otherwise.

Public API (also surfaced as ``daedalus.plot_prediagrams``):
    plot_prediagrams(model, k, max_ell, save=None, ncol=4) -> matplotlib Figure
    contributing_prediagrams(model, k, max_ell) -> {ell: [prediagram, ...]}
"""
import collections, string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

XS, YS = 2.9, 1.9   # layer (x) and within-layer (y) spacing


def _roman(n):
    out, vals = '', [(10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')]
    for v, s in vals:
        while n >= v: out += s; n -= v
    return out


def contributing_prediagrams(model, k, max_ell):
    """The prediagrams (filtered directed topologies) contributing to the
    ``k``-point cumulant up to loop order ``max_ell`` for ``model`` (a model
    dict from :func:`daedalus.load_model`).  Returns ``{ell: [prediagram]}``
    where each prediagram is the ``(D, G, leaves, internal)`` tuple from the
    enumerator (``D`` a Sage DiGraph)."""
    from engine.core.field_theory import FieldTheory
    from engine.core.vertices import extract_vertex_types, extract_source_types
    from engine.enumeration.loop_diagram_enumeration import enumerate_all as enum_pre
    from engine.diagrams.filter import filter_prediagrams
    ft = FieldTheory(model, taylor_order=max(k + 2 * max_ell, 2)); ft.expand()
    vt = extract_vertex_types(ft); st = extract_source_types(ft)
    out = {}
    for ell in range(max_ell + 1):
        _, _, pres, _ = enum_pre(k, ell, verbose=False)
        kept, _ = filter_prediagrams(pres, vt, st)
        out[ell] = kept
    return out


# ── geometry helpers ──────────────────────────────────────────────────
def _seg_int(p, q, r, s):
    def o(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = o(r, s, p), o(r, s, q), o(p, q, r), o(p, q, s)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _edge_rads(pos, edges):
    """arc3 curvature for every edge copy — the SINGLE source of truth shared by
    the drawing and the layout optimiser, so the optimiser scores exactly what
    gets drawn.  Parallel edges fan into a lens; a single edge stays straight
    unless its segment passes too close to another node, in which case it bows
    away from the offenders (rad sign: positive pushes the apex to the right of
    travel u→v, i.e. toward −(perp) since the control point is mid + rad·(dy,−dx))."""
    xcols = sorted(set(round(pos[v][0], 2) for v in pos))
    rank_of = {x: i for i, x in enumerate(xcols)}

    def _rank(v):
        return rank_of[round(pos[v][0], 2)]

    pair = collections.defaultdict(list)
    for e in edges:
        pair[frozenset((e[0], e[1]))].append(e)
    rads = {}
    for key, es in pair.items():
        m = len(es)
        for i, e in enumerate(sorted(es, key=lambda e: str(e[2]))):
            u, v, _ = e
            if m > 1:                                    # parallel edges -> lens bubble
                # lens half-width = rad*L/2; 0.27 keeps the two arcs clearly
                # separate but much slimmer than the old 0.42 (user feedback)
                rads[e] = 0.27 * (i - (m - 1) / 2.0) * 2
                continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ddx, ddy = x1 - x0, y1 - y0
            seg = ddx * ddx + ddy * ddy
            side, blockers = 0.0, []
            for w in pos:                                # ANY nearby node blocks
                if w == u or w == v:
                    continue
                xw, yw = pos[w]
                t = 0.5 if seg < 1e-12 else ((xw - x0) * ddx + (yw - y0) * ddy) / seg
                if not (0.04 < t < 0.96):
                    continue
                d = ((xw - (x0 + t * ddx)) ** 2 + (yw - (y0 + t * ddy)) ** 2) ** 0.5
                if d < 0.34:
                    ss = ddx * (yw - y0) - ddy * (xw - x0)    # >0: w left of u->v
                    blockers.append((t, d, ss))
                    # weight by 1/d: bow away from the CLOSEST blocker; without
                    # this, blockers on opposite sides cancel and the edge runs
                    # dead straight THROUGH one of them
                    side += (1.0 if ss > 0 else -1.0) / max(d, 0.05)
            if not blockers:
                rads[e] = 0.0
            else:
                # ADAPTIVE curvature: the quad-bezier offset from the chord at
                # parameter t is 2t(1-t)*rad*L, so bow with just enough rad that
                # every blocker on the fled side clears by >= 0.36 -- long edges
                # get a gentle sweep instead of a fixed bulge, short dodges stay
                # tight.  Blockers on the approached side cap the bow (squeeze ->
                # compromise midway); the untangler's node-on-curve term steers
                # layouts away from genuinely squeezed configurations.
                L = seg ** 0.5
                sgn = 1.0 if side >= 0 else -1.0      # rad>0: apex RIGHT of travel
                lo, hi = 0.12, 0.45
                for (t, d, ss) in blockers:
                    f = 2.0 * t * (1.0 - t) * L
                    if f < 1e-9:
                        continue
                    if (ss > 0) == (sgn > 0):          # fled side: must clear it
                        lo = max(lo, (0.36 - d) / f)
                    else:                              # approached side: don't hit it
                        hi = min(hi, max((d - 0.30) / f, 0.0))
                mag = min(lo, 0.45) if hi >= lo else max(0.12, min(0.45, (lo + hi) / 2.0))
                rads[e] = mag * sgn
    return rads


def _edge_poly(pos, e, rad, ts=(0.0, 1 / 3.0, 2 / 3.0, 1.0)):
    """Sampled points of the drawn (quadratic-bezier) edge curve."""
    u, v = e[0], e[1]
    (x0, y0), (x1, y1) = pos[u], pos[v]
    dx, dy = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2.0 + rad * dy
    cy = (y0 + y1) / 2.0 - rad * dx
    out = []
    for t in ts:
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
        out.append((a * x0 + b * cx + c * x1, a * y0 + b * cy + c * y1))
    return out


def _pt_seg_d2(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _roles(D, leaves):
    from engine.diagrams.filter import classify_prediagram_vertices
    s, i = classify_prediagram_vertices(D, leaves)
    return [int(v) for v in leaves], [int(v) for v in s], [int(v) for v in i]


def _layout_graphviz(D, leaves):
    """Use graphviz 'dot' (layered layout: crossing-min + dummy-node long-edge
    routing) for node positions, in the right->left time convention.  Returns
    the same tuple as the hand-rolled fallback so draw_prediagram is unchanged."""
    import pydot
    nodes = [int(v) for v in D.vertices()]
    edges = [(int(u), int(v), lbl) for (u, v, lbl) in D.edges()]
    ext_v, src_v, int_v = _roles(D, leaves)
    g = pydot.Dot(graph_type='digraph', rankdir='RL', nodesep='0.7', ranksep='1.05')
    for n in nodes:
        g.add_node(pydot.Node(str(n), shape='point', width='0.12'))
    # Pin the roles to their conventional columns (sources rightmost = min rank,
    # externals leftmost = max rank).  Without this, dot may park a source in an
    # internal column or an external mid-figure, producing role-mixed columns
    # that read badly and that the _untangle column permutation cannot fix.
    if src_v:
        sg = pydot.Subgraph(rank='min')
        for n in src_v: sg.add_node(pydot.Node(str(n)))
        g.add_subgraph(sg)
    if ext_v:
        sg = pydot.Subgraph(rank='max')
        for n in ext_v: sg.add_node(pydot.Node(str(n)))
        g.add_subgraph(sg)
    seen = set()
    for u, v, _ in edges:
        if u != v and (u, v) not in seen:
            g.add_edge(pydot.Edge(str(u), str(v))); seen.add((u, v))
    raw = g.create_dot()
    parsed = pydot.graph_from_dot_data(raw.decode() if isinstance(raw, bytes) else raw)[0]
    pos = {}
    for node in parsed.get_nodes():
        nm = node.get_name().strip('"')
        if not nm.lstrip('-').isdigit():
            continue
        p = node.get_pos()
        if p:
            x, y = map(float, p.strip('"').split(','))
            pos[int(nm)] = [x, y]
    if len(pos) < len(nodes):
        raise RuntimeError('graphviz layout incomplete')
    # orient: sources to the right, externals to the left
    if src_v and ext_v and np.mean([pos[v][0] for v in src_v]) < np.mean([pos[v][0] for v in ext_v]):
        for v in pos: pos[v][0] = -pos[v][0]
    # normalize so the median nearest-neighbour gap ~= 2.0 data units
    P = np.array([pos[v] for v in nodes])
    gaps = []
    for i in range(len(P)):
        d = np.hypot(*(P - P[i]).T); d[i] = np.inf
        gaps.append(d.min())
    sc = 2.0 / (np.median(gaps) or 1.0)
    # stretch y by 1.85: graphviz packs rows tightly; the extra vertical air
    # separates fan-outs and (with per-row panel heights) fills the cell —
    # a taller total figure is preferred over cramped prediagrams
    for v in pos: pos[v] = [pos[v][0] * sc, pos[v][1] * sc * 1.85]
    return pos, ext_v, src_v, int_v, edges


def _untangle(pos, edges):
    """Deterministic layout post-pass, in two stages.

    Stage 1 — slot permutation: within each x-column, permute which node sits
    on which of the column's y-slots (exact joint search when the space is
    small, per-column coordinate descent otherwise).  This is the "permute the
    sources" freedom.
    Stage 2 — y-nudging: a deterministic local search that moves individual
    nodes OFF the slots vertically, e.g. lifting the far vertex of a bubble
    clear of a crowded corridor.

    Both stages score the geometry that is ACTUALLY drawn — bezier arcs via
    the shared :func:`_edge_rads` (bubble lenses, blocker bows), counting in
    order (a) curve–curve crossings, (b) nodes lying on a foreign edge curve,
    (c) edges forced to arc, and (d) total edge length.  Role labels are
    assigned from the FINAL y-positions by :func:`_generic_labels`, and the
    mapping table re-derives them through this same deterministic pipeline, so
    figure and label map stay consistent by construction."""
    from itertools import permutations, product
    from math import factorial
    cols = collections.defaultdict(list)                 # x-column -> nodes
    for v in pos:
        cols[round(pos[v][0], 2)].append(v)
    xs_sorted = sorted(cols)
    colnodes = [sorted(cols[x]) for x in xs_sorted]      # deterministic order
    slots = [sorted((pos[v][1] for v in cols[x])) for x in xs_sorted]
    ecopies = sorted((e for e in edges if e[0] != e[1]),
                     key=lambda e: (e[0], e[1], str(e[2])))

    def cost(p):
        rads = _edge_rads(p, ecopies)
        polys = [(_edge_poly(p, e, rads[e]), e) for e in ecopies]
        cross = 0
        for a in range(len(polys)):
            pa, ea = polys[a]
            for b in range(a + 1, len(polys)):
                pb, eb = polys[b]
                if len({ea[0], ea[1], eb[0], eb[1]}) < 4:
                    continue                              # shared endpoint: skip
                hit = False
                for i in range(len(pa) - 1):
                    for j in range(len(pb) - 1):
                        if _seg_int(pa[i], pa[i + 1], pb[j], pb[j + 1]):
                            hit = True; break
                    if hit: break
                if hit: cross += 1
        node_hits, arcs, L2 = 0, 0, 0.0
        for poly, e in polys:
            u, v = e[0], e[1]
            L2 += (p[v][0] - p[u][0]) ** 2 + (p[v][1] - p[u][1]) ** 2
            if abs(rads[e]) > 1e-9:
                arcs += 1
            for w in p:                                   # node sitting on a foreign curve
                if w == u or w == v:
                    continue
                d2 = min(_pt_seg_d2(p[w], poly[i], poly[i + 1])
                         for i in range(len(poly) - 1))
                if d2 < 0.38 * 0.38:
                    node_hits += 1
        vs = sorted(p)                                    # crowded node pairs (any column)
        crowd = sum(1 for i in range(len(vs)) for j in range(i + 1, len(vs))
                    if (p[vs[i]][0] - p[vs[j]][0]) ** 2
                    + (p[vs[i]][1] - p[vs[j]][1]) ** 2 < 0.85 * 0.85)
        return (cross, node_hits, arcs, crowd, round(L2, 6))

    def build(assign_per_col):
        p = {}
        for ci, order in enumerate(assign_per_col):
            for si, v in enumerate(order):
                p[v] = [pos[v][0], slots[ci][si]]
        return p

    # ── stage 1: slot permutations ────────────────────────────────────
    total = 1
    for ns in colnodes:
        total *= factorial(len(ns))
    if total <= 2000:                                    # small: exact joint search
        best, best_c = None, None
        for combo in product(*(list(permutations(ns)) for ns in colnodes)):
            p = build(combo); c = cost(p)
            if best_c is None or c < best_c:
                best, best_c = p, c
    else:                                                # deterministic coord. descent
        cur = [list(ns) for ns in colnodes]
        best = build(cur); best_c = cost(best)
        for _sweep in range(6):
            improved = False
            for ci in range(len(cur)):
                for perm in permutations(sorted(cur[ci])):
                    trial = list(cur); trial[ci] = list(perm)
                    p = build(trial); c = cost(p)
                    if c < best_c:
                        best, best_c, cur = p, c, trial; improved = True
            if not improved:
                break
    # ── stage 2: vertical nudging off the slots ───────────────────────
    ylo = min(y for ys in slots for y in ys) - 1.6
    yhi = max(y for ys in slots for y in ys) + 1.6
    for _sweep in range(3):
        improved = False
        for v in sorted(best):
            same_col = [w for w in best
                        if w != v and round(best[w][0], 2) == round(best[v][0], 2)]
            y0 = best[v][1]
            for dy in (-1.4, -1.05, -0.7, -0.35, 0.35, 0.7, 1.05, 1.4):
                yn = y0 + dy
                if not (ylo <= yn <= yhi):
                    continue
                if any(abs(best[w][1] - yn) < 0.9 for w in same_col):
                    continue                              # keep same-column spacing
                best[v][1] = yn
                c = cost(best)
                if c < best_c:
                    best_c = c; y0 = yn; improved = True
                else:
                    best[v][1] = y0
        if not improved:
            break
    return best


def _layout(D, leaves):
    try:
        pos, ext_v, src_v, int_v, edges = _layout_graphviz(D, leaves)
    except Exception:
        pos, ext_v, src_v, int_v, edges = _layout_layered(D, leaves)
    pos = _untangle(pos, edges)
    return pos, ext_v, src_v, int_v, edges


def _layout_layered(D, leaves):
    nodes = [int(v) for v in D.vertices()]
    edges = [(int(u), int(v), lbl) for (u, v, lbl) in D.edges()]
    ext_v, src_v, int_v = _roles(D, leaves)
    # cycle-robust depth via SCC condensation: bubbles are 2-cycles (a<->b),
    # which never reach in-deg 0 and would collapse onto the sources (depth 0).
    from collections import deque
    try:
        sccs = [[int(v) for v in comp] for comp in D.strongly_connected_components()]
    except Exception:
        sccs = [[v] for v in nodes]
    scc_of = {v: i for i, comp in enumerate(sccs) for v in comp}
    csucc = collections.defaultdict(set); cindeg = collections.defaultdict(int)
    cnodes = set(range(len(sccs)))
    for u, v, _ in edges:
        a, b = scc_of[u], scc_of[v]
        if a != b and b not in csucc[a]:
            csucc[a].add(b); cindeg[b] += 1
    q = deque([c for c in cnodes if cindeg[c] == 0]); cdepth = {c: 0 for c in cnodes}; ci = dict(cindeg)
    while q:
        u = q.popleft()
        for v in csucc[u]:
            cdepth[v] = max(cdepth[v], cdepth[u] + 1); ci[v] -= 1
            if ci[v] == 0: q.append(v)
    depth = {v: cdepth[scc_of[v]] for v in nodes}
    if ext_v:
        md = max(depth.values())
        for v in ext_v: depth[v] = md
    maxd = max(depth.values()) if depth else 0
    layers = collections.defaultdict(list)
    for v in sorted(nodes): layers[depth[v]].append(v)

    simple = [(u, v) for (u, v, _) in edges if u != v]   # for crossing count

    def assign():
        # backbone: sources (d=0) and externals (d=maxd) fan out vertically;
        # internal layers hug the spine (compact y) so they read as a horizontal chain.
        pos = {}
        for d, vs in layers.items():
            endcap = (d == 0) or (d == maxd)
            sp = YS if endcap else YS * 0.95
            for i, v in enumerate(vs):
                pos[v] = (XS * (maxd - d), (i - (len(vs) - 1) / 2.0) * sp)
        return pos

    def crossings(pos):
        c = 0
        for a in range(len(simple)):
            u1, v1 = simple[a]
            for b in range(a + 1, len(simple)):
                u2, v2 = simple[b]
                if len({u1, v1, u2, v2}) < 4: continue
                if _seg_int(pos[u1], pos[v1], pos[u2], pos[v2]): c += 1
        return c

    nbr = collections.defaultdict(list)
    for u, v, _ in edges:
        if u != v: nbr[u].append(v); nbr[v].append(u)
    best_order = {d: list(vs) for d, vs in layers.items()}
    best_c = crossings(assign())
    for sweep in range(10):
        # median reorder within each layer
        for d in layers:
            if len(layers[d]) < 2: continue
            pos = assign()
            layers[d].sort(key=lambda v: np.median([pos[w][1] for w in nbr[v]]) if nbr[v] else pos[v][1])
        # transpose: adjacent swaps that reduce crossings
        improved = True
        while improved:
            improved = False
            for d in layers:
                vs = layers[d]
                for i in range(len(vs) - 1):
                    vs[i], vs[i + 1] = vs[i + 1], vs[i]
                    c = crossings(assign())
                    if c < best_c:
                        best_c = c; best_order = {dd: list(v2) for dd, v2 in layers.items()}; improved = True
                    else:
                        vs[i], vs[i + 1] = vs[i + 1], vs[i]
        layers = {d: list(v) for d, v in best_order.items()}
    pos = assign()
    # backbone: straighten the longest ext->src path to y≈0 (nicer spine)
    return pos, ext_v, src_v, int_v, edges


# ── topology signature for grouping ───────────────────────────────────
def topo_signature(D, leaves):
    ext_v, src_v, int_v = _roles(D, leaves)
    mult = collections.Counter()
    for (u, v, _) in D.edges(): mult[frozenset((int(u), int(v)))] += 1
    # full multi-edge profile (one entry per vertex pair with multiplicity > 1,
    # descending), so e.g. one bubble and two bubbles land in DIFFERENT groups.
    # The signature carries exactly what the group header shows — grouping any
    # finer (e.g. by internal degree profile) splits groups whose headers read
    # identically, which displays as confusing duplicate headings.
    mults = tuple(sorted((m for m in mult.values() if m > 1), reverse=True))
    return (len(int_v), len(src_v), mults)


def _sig_sort_key(sig):
    """Deterministic group order, shared by the figure and the mapping table:
    fewer internals first, then fewer sources, then richer multi-edge motifs
    (higher max multiplicity, then more of them)."""
    n_int, n_src, mults = sig
    return (n_int, n_src, -(mults[0] if mults else 1), -len(mults), mults)


def _grouped(pre):
    """``[(ell, sig, [prediagram, ...])]`` in the shared display order — the
    SINGLE grouping used by both the figure and the mapping table, so their
    group order and #index numbering agree structurally."""
    out = []
    for ell in sorted(pre):
        bysig = collections.OrderedDict()
        for pd in pre[ell]:
            bysig.setdefault(topo_signature(pd[0], pd[2]), []).append(pd)
        for sig in sorted(bysig, key=_sig_sort_key):
            out.append((ell, sig, bysig[sig]))
    return out


def sig_label(sig):
    n_int, n_src, mults = sig
    c = collections.Counter(mults)
    parts = []
    for m in sorted(c, reverse=True):
        name = {2: 'bubble', 3: 'sunset'}.get(m, '%d-edge' % m)
        parts.append(name if c[m] == 1 else '%d %ss' % (c[m], name))
    motif = (' · ' + ' + '.join(parts)) if parts else ''
    return '%d internal, %d source%s%s' % (n_int, n_src, '' if n_src == 1 else 's', motif)


# ── draw one prediagram ───────────────────────────────────────────────
_SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def _generic_labels(D, leaves):
    """Layout + generic role labels for a prediagram, shared by the figure and
    the mapping table so they agree exactly.  A clean, role-distinct taxonomy:

        sources           -> i, ii, iii, …   (roman numerals, right)
        internal vertices -> a, b, c, …      (latin letters, middle)
        external legs     -> 1, 2, …         (arabic numerals, left)

    all ordered top→bottom.  Propagators are NOT given their own symbol; each is
    named by its endpoints, e.g. ``a→b`` (direction distinguishes the two halves
    of a bubble), with a subscript only to separate same-direction parallel
    edges.  Returns (pos, ext_v, src_v, int_v, edges, srclab, intlab, extlab,
    ext_order, edge_name)."""
    pos, ext_v, src_v, int_v, edges = _layout(D, leaves)
    srclab = {v: _roman(i + 1) for i, v in enumerate(sorted(src_v, key=lambda v: -pos[v][1]))}
    intlab = {v: string.ascii_lowercase[i] for i, v in enumerate(sorted(int_v, key=lambda v: -pos[v][1]))}
    ext_order = sorted(ext_v, key=lambda v: -pos[v][1])      # top -> bottom
    extlab = {v: str(i + 1) for i, v in enumerate(ext_order)}
    vlab = {**srclab, **intlab, **extlab}
    by_dir = collections.defaultdict(list)
    for e in edges:
        by_dir[(e[0], e[1])].append(e)
    edge_name = {}
    for (u, w), es in by_dir.items():
        base = '%s→%s' % (vlab.get(u, '?'), vlab.get(w, '?'))
        if len(es) == 1:
            edge_name[es[0]] = base
        else:                                # same-direction parallel edges -> a→b₁, a→b₂
            for j, e in enumerate(sorted(es, key=lambda e: e[2])):
                edge_name[e] = base + str(j + 1).translate(_SUB)
    return pos, ext_v, src_v, int_v, edges, srclab, intlab, extlab, ext_order, edge_name


def draw_prediagram(D, leaves, ax, title=None, labels=None):
    if labels is None:
        labels = _generic_labels(D, leaves)
    pos, ext_v, src_v, int_v, edges, srclab, intlab, extlab, ext_order, _ = labels
    BLACK, EDGEC = '#181818', '#3a3a3a'
    track = [list(pos[v]) for v in pos]              # extent: nodes + arc apexes + labels
    # Curvatures from the SAME helper the layout optimiser scored, so what was
    # optimised is exactly what is drawn.
    rads = _edge_rads(pos, edges)
    dense = []                                       # sampled curves (label placement)
    for e in sorted(rads, key=lambda e: (e[0], e[1], str(e[2]))):
        u, v = e[0], e[1]
        rad = rads[e]
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        mx, my = (pos[u][0] + pos[v][0]) / 2.0, (pos[u][1] + pos[v][1]) / 2.0
        cx, cy = mx + rad * dy, my - rad * dx        # arc3 control point
        P0, P1 = pos[u], pos[v]
        def _bez(t, P0=P0, P1=P1, cx=cx, cy=cy):     # point on the (curved) edge
            a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
            return (a * P0[0] + b * cx + c * P1[0], a * P0[1] + b * cy + c * P1[1])
        # The edge line runs to the vertex CENTRES (shrink 0) so it physically reaches
        # each node; the filled/hollow circles are drawn on top (higher zorder) and
        # hide the overshoot, so the line emerges right at the vertex boundary.
        ax.add_patch(FancyArrowPatch(P0, P1, connectionstyle=f'arc3,rad={rad}',
                     arrowstyle='-', lw=1.5, color=EDGEC, shrinkA=0, shrinkB=0,
                     zorder=1, joinstyle='round'))
        # Arrowhead at the MIDDLE of the edge (not the end), pointing along the flow
        # u -> v.  Drawn between two nearby points on the curve so it follows the arc.
        ax.add_patch(FancyArrowPatch(_bez(0.42), _bez(0.56), arrowstyle='-|>',
                     mutation_scale=13, lw=1.5, color=EDGEC, shrinkA=0, shrinkB=0,
                     zorder=2))
        track.append(list(_bez(0.5)))                # arc apex (for the limits)
        dense.append((e, [_bez(t / 8.0) for t in range(9)]))

    # ── collision-aware label placement ──────────────────────────────
    # Try the conventional anchor first (internals above, sources right,
    # externals left); if an edge curve or another node sits on it, walk the
    # candidate list and take the first clear spot (else the clearest).
    placed = []                                       # already-placed label anchors
    def _clearance(p, skip):
        d2 = 9.0
        for e, pts in dense:
            # skip the segment where an edge emerges from under the label's own
            # node disc — being near one's own departure point is not occlusion
            i0 = 1 if e[0] == skip else 0
            i1 = (len(pts) - 2) if e[1] == skip else (len(pts) - 1)
            for i in range(i0, i1):
                d2 = min(d2, _pt_seg_d2(p, pts[i], pts[i + 1]))
        for w in pos:
            if w == skip:
                continue
            d2 = min(d2, (p[0] - pos[w][0]) ** 2 + (p[1] - pos[w][1]) ** 2)
        for q in placed:                              # labels must not overlap each other
            d2 = min(d2, ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) * 0.4)
        return d2
    def _glyph_center(p, ha, va):
        # ha/va offset the glyph box from the anchor; clearance must be judged
        # where the TEXT actually sits (e.g. va='top' text grows downward)
        return (p[0] + (0.11 if ha == 'left' else -0.11 if ha == 'right' else 0.0),
                p[1] + (0.14 if va == 'bottom' else -0.14 if va == 'top' else 0.0))
    def _place(v, cands, clear=0.30):
        best = None
        for (dxo, dyo, ha, va) in cands:
            p = (pos[v][0] + dxo, pos[v][1] + dyo)
            pc = _glyph_center(p, ha, va)
            c2 = _clearance(pc, v)
            if c2 >= clear * clear:
                placed.append(pc)
                return p, ha, va
            if best is None or c2 > best[0]:
                best = (c2, p, ha, va, pc)
        placed.append(best[4])
        return best[1], best[2], best[3]
    C_INT = [(0.0, 0.40, 'center', 'bottom'), (0.0, -0.42, 'center', 'top'),
             (-0.32, 0.32, 'right', 'bottom'), (0.32, 0.32, 'left', 'bottom'),
             (-0.32, -0.34, 'right', 'top'), (0.32, -0.34, 'left', 'top'),
             (-0.40, 0.0, 'right', 'center'), (0.40, 0.0, 'left', 'center'),
             # escape ring, when every near anchor is blocked
             (0.0, 0.66, 'center', 'bottom'), (0.0, -0.68, 'center', 'top'),
             (-0.52, 0.52, 'right', 'bottom'), (0.52, 0.52, 'left', 'bottom'),
             (-0.52, -0.54, 'right', 'top'), (0.52, -0.54, 'left', 'top')]
    C_SRC = [(0.34, 0.0, 'left', 'center'), (0.32, 0.30, 'left', 'bottom'),
             (0.32, -0.32, 'left', 'top'), (0.0, 0.40, 'center', 'bottom'),
             (0.0, -0.42, 'center', 'top')]
    C_EXT = [(-0.34, 0.0, 'right', 'center'), (-0.32, 0.30, 'right', 'bottom'),
             (-0.32, -0.32, 'right', 'top')]
    # Propagators carry no symbol of their own -- they are named by their endpoints
    # (a→b) in the mapping table, so the figure stays uncluttered.
    for v in ext_v:                                        # external legs: hollow, labelled 1,2,…
        ax.add_patch(Circle(pos[v], 0.16, fc='white', ec=BLACK, lw=1.8, zorder=3))
        p, ha, va = _place(v, C_EXT)
        ax.text(p[0], p[1], extlab[v], ha=ha, va=va, fontsize=10.5, zorder=4)
        track.append([p[0] - (0.28 if ha == 'right' else 0.0), p[1]])
    for v in src_v:                                        # noise sources: filled SQUARE
        s = 0.155                                          # (distinct from circular vertices/legs)
        ax.add_patch(Rectangle((pos[v][0] - s, pos[v][1] - s), 2 * s, 2 * s,
                               fc=BLACK, ec=BLACK, zorder=3))
        p, ha, va = _place(v, C_SRC)
        ax.text(p[0], p[1], srclab[v], ha=ha, va=va, fontsize=11, fontstyle='italic', zorder=4)
        track.append([p[0] + (0.34 if ha == 'left' else 0.0), p[1]])
    for v in int_v:
        ax.add_patch(Circle(pos[v], 0.17, fc=BLACK, ec=BLACK, zorder=3))
        p, ha, va = _place(v, C_INT)
        ax.text(p[0], p[1], intlab[v], ha=ha, va=va, fontsize=11, fontweight='bold', zorder=4)
        track.append([p[0], p[1] + (0.26 if va == 'bottom' else -0.26 if va == 'top' else 0.0)])
    # Limits enclose nodes, arc apexes AND every label, so nothing clips and the
    # title clears the topmost vertex label; small uniform margin beyond that.
    xs = [p[0] for p in track]; ys = [p[1] for p in track]
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.3); ax.set_ylim(min(ys) - 0.3, max(ys) + 0.32)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=10, pad=5)


# ── grouped figure (public entry point) ───────────────────────────────
def plot_prediagrams(model, k, max_ell, save=None, ncol=None):
    """Draw the contributing prediagrams for ``model`` at correlator order
    ``k`` and loop order ``max_ell``, grouped by topology family and labelled
    by role (sources i,ii right; internals a,b,c middle; external legs 1,2 left;
    propagators named by endpoints, e.g. a→b).  ``model`` is a model dict (from
    :func:`daedalus.load_model`) or a model-name string.  Returns the
    matplotlib ``Figure``; also writes a PNG when ``save`` is given."""
    from matplotlib.gridspec import GridSpec
    if isinstance(model, str):
        import daedalus as dd
        name = model; model, _ = dd.load_model(model)
    else:
        name = model.get('name', 'model')
    pre = contributing_prediagrams(model, k, max_ell)
    groups = _grouped(pre)   # (ell, sig, [prediagrams]) — shared with mappings
    # Adaptive sizing: k≥3 diagrams carry many external legs + sources (more
    # fan-out and crossings), so render them with fewer-per-row + larger cells.
    if ncol is None:
        ncol = 2          # 2-up: each prediagram gets half the figure width
                          # (a longer figure is preferred over small panels)
    PANELW = 4.3 if k <= 2 else 5.6
    # build a row plan: a thin header row per group, then diagram rows.
    # Layouts are computed ONCE here (reused by draw_prediagram below) so each
    # row's height can match its tallest diagram -- a flat chain gets a short
    # row, a stacked 2-loop diagram a tall one, killing the letterbox whitespace
    # that fixed-height cells produced under aspect='equal'.
    HEAD = 0.5
    plan = []   # ('head', text) | ('row', [(pd, labels)], c, row_h)
    for (ell, sig, pds) in groups:
        plan.append(('head', r'$\ell=%d$   ·   %s   ·   %d diagram%s'
                     % (ell, sig_label(sig), len(pds), '' if len(pds) == 1 else 's')))
        labs = [_generic_labels(pd[0], pd[2]) for pd in pds]
        for c in range(0, len(pds), ncol):
            row = list(zip(pds[c:c + ncol], labs[c:c + ncol]))
            row_h = 2.4                                # generous floor: even flat
            for _pd, lb in row:                        # chains get a roomy panel
                xs = [p[0] for p in lb[0].values()]; ys = [p[1] for p in lb[0].values()]
                w = (max(xs) - min(xs)) + 2.1          # side labels + margins
                h = (max(ys) - min(ys)) + 1.8          # top/bottom labels + title
                row_h = max(row_h, PANELW * h / max(w, 1e-9))
            plan.append(('row', row, c, min(row_h, 7.5)))   # c = index of the row's
                                                            # first diagram in its group
    if not plan:                          # nothing contributes at this (k, ell)
        fig = plt.figure(figsize=(6.5, 1.8))
        fig.text(0.5, 0.5, 'No contributing prediagrams for %s at k=%d, ℓ≤%d'
                 % (name, k, max_ell), ha='center', va='center', fontsize=12)
        if save:
            fig.savefig(save, dpi=145, bbox_inches='tight')
        plt.close(fig)        # drop from pyplot's registry so the inline backend
        return fig            # doesn't auto-show it (the returned fig renders once)
    heights = [HEAD if p[0] == 'head' else p[3] for p in plan]
    fig = plt.figure(figsize=(PANELW * ncol, sum(heights) + 0.8))
    gs = GridSpec(len(plan), ncol, height_ratios=heights, hspace=0.26, wspace=0.16,
                  left=0.02, right=0.98, top=1 - 0.5 / (sum(heights) + 0.8), bottom=0.4 / (sum(heights) + 0.8))
    for ri, p in enumerate(plan):
        if p[0] == 'head':
            ax = fig.add_subplot(gs[ri, :]); ax.axis('off')
            ax.add_patch(plt.Rectangle((0, 0.12), 1, 0.76, transform=ax.transAxes, fc='#eef2f8', ec='none', zorder=0))
            ax.text(0.012, 0.5, p[1], transform=ax.transAxes, fontsize=11, fontweight='bold',
                    color='#1c3a66', va='center', ha='left')
        else:
            for ci, (pd, lb) in enumerate(p[1]):
                ax = fig.add_subplot(gs[ri, ci])
                # continuous numbering within the group (#1..#N across rows),
                # matching the #idx of the prediagram_mappings table exactly
                draw_prediagram(pd[0], pd[2], ax, title='#%d' % (p[2] + ci + 1),
                                labels=lb)
    fig.suptitle('Contributing prediagrams: %s,  k=%d, ℓ≤%d' % (name, k, max_ell),
                 fontsize=13, y=0.998)
    # pin the legend 0.12in above the bottom edge (a FRACTION would drift into
    # the last row as the figure grows taller)
    fig.text(0.5, 0.12 / (sum(heights) + 0.8), r'time $\leftarrow$      $\circ\;1,2$ external legs      '
             r'$\blacksquare\;i,ii$ sources      $\bullet\;a,b,c$ internal vertices      '
             r'(propagators named by endpoints, e.g. $a\to b$)',
             ha='center', fontsize=10, color='#555')
    if save:
        fig.savefig(save, dpi=145, bbox_inches='tight')
    plt.close(fig)            # drop from pyplot's registry so the inline backend
    return fig                # doesn't auto-show it (the returned fig renders once)


# ── label-mapping tables (generic label -> typed-diagram field types) ─────────

def _leg_str(leg):
    """Render a (field_base, pop_idx) leg, e.g. ('dn', 2) -> 'dn_2'."""
    if leg is None:
        return '?'
    base, pop = leg
    return '%s_%d' % (base, int(pop)) if (pop and int(pop) > 1) else str(base)


def _coeff_str(c):
    s = str(c)
    return s if len(s) <= 26 else s[:25] + '…'


def diagram_label_map(td, labels):
    """Map one typed diagram's generic labels to its field types.  ``labels`` is
    the tuple from :func:`_generic_labels` for the SAME prediagram, so the
    indices line up with the figure.  Returns a dict with keys 'sources',
    'internals', 'propagators', 'externals' (lists of rows)."""
    _, _, _, _, _, srclab, intlab, extlab, ext_order, edge_name = labels
    # normalise keys to plain ints / int-tuples to match the figure's labels
    va = {int(v): a for v, a in td.vertex_assignments.items()}
    et = {(int(u), int(v), lbl): rp for (u, v, lbl), rp in td.edge_types.items()}
    el = {int(v): f for v, f in td.external_legs.items()}
    out = {'sources': [], 'internals': [], 'propagators': [], 'externals': []}
    for v in sorted(srclab, key=lambda v: srclab[v]):
        a = va.get(v)
        n = int(a.bigrade[0]) if a is not None else 0
        legs = ' '.join(_leg_str(l) for l in (a.response_legs if a else []))
        out['sources'].append((srclab[v], 'K^(%d)' % n, legs))
    for v in sorted(intlab, key=lambda v: intlab[v]):
        a = va.get(v)
        if a is None:
            out['internals'].append((intlab[v], '?', '')); continue
        resp = ' '.join(_leg_str(l) for l in getattr(a, 'response_legs', []))
        phys = ' '.join(_leg_str(l) for l in getattr(a, 'physical_legs', []))
        out['internals'].append((intlab[v], _coeff_str(a.coefficient),
                                 'resp⟨%s⟩ phys⟨%s⟩' % (resp, phys)))
    for e in sorted(edge_name, key=lambda e: edge_name[e]):
        rp = et.get(e)
        if rp is None:
            out['propagators'].append((edge_name[e], '?')); continue
        resp_leg, phys_leg = rp
        out['propagators'].append(
            (edge_name[e], 'G[%s ← %s]' % (_leg_str(phys_leg), _leg_str(resp_leg))))
    for leaf in ext_order:
        out['externals'].append((extlab[leaf], _leg_str(el.get(leaf))))
    return out


def prediagram_mappings(model, k, max_ell, external_fields=None,
                        use_propagator=False, max_typings=6, printout=True):
    """For each contributing prediagram, list its typed realizations as
    label-maps (generic prediagram label -> field type), matching the figure
    from :func:`plot_prediagrams`.

    Returns ``(result, text)``: ``result`` is ``{ell: [entry, ...]}`` (each
    entry has 'sig', 'idx', 'labels', 'typings'); ``text`` is a printable
    rendering.  ``max_typings`` caps how many typings are *printed* per
    prediagram (all are kept in ``result``).  ``use_propagator=True`` builds the
    propagator and prunes identically-zero typings (the exact contributing
    set); the default skips that build and may list a few extra zero typings."""
    if isinstance(model, str):
        import daedalus as dd
        name = model; model, _ = dd.load_model(model)
    else:
        name = model.get('name', 'model')
    from engine.core.field_theory import FieldTheory
    from engine.core.vertices import extract_vertex_types, extract_source_types
    from engine.diagrams.type_assignment import (enumerate_typed_diagrams,
                                                build_field_index_map)
    ft = FieldTheory(model, taylor_order=max(k + 2 * max_ell, 2)); ft.expand()
    vt = extract_vertex_types(ft); st = extract_source_types(ft)
    ri, pi = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    if external_fields is None:
        pf = model.get('physical_fields') or []
        base = pf[0]['name'] if pf else 'phi'
        external_fields = [(base, 1)] * k
    external_fields = [tuple(e) for e in external_fields]
    G_ft = None
    if use_propagator:
        from api._propagator import build_propagator
        G_ft = build_propagator(ft, model, verbose=False)['G_ft']
    pre = contributing_prediagrams(model, k, max_ell)
    result = {}
    for (ell, sig, pds) in _grouped(pre):     # same groups/order as the figure
        for idx, pd in enumerate(pds):
            labels = _generic_labels(pd[0], pd[2])
            typings = [diagram_label_map(td, labels)
                       for td in enumerate_typed_diagrams(
                           pd, external_fields, vt, st, G_ft, ri, pi)]
            result.setdefault(ell, []).append({'sig': sig, 'idx': idx + 1,
                                               'labels': labels, 'typings': typings})
    for ell in pre:
        result.setdefault(ell, [])
    text = _format_mappings(name, k, max_ell, result, max_typings)
    if printout:
        print(text)
    return result, text


def _format_mappings(name, k, max_ell, result, max_typings=6):
    L = ['═' * 72,
         ' %s — prediagram → typed-diagram label maps  (k=%d, ℓ≤%d)'
         % (name, k, max_ell), '═' * 72]
    for ell in sorted(result):
        for e in result[ell]:
            nt = len(e['typings'])
            L.append('')
            L.append('ℓ=%d · %s · #%d   (%d typing%s)'
                     % (ell, sig_label(e['sig']), e['idx'], nt,
                        '' if nt == 1 else 's'))
            for ti, tm in enumerate(e['typings'][:max_typings]):
                L.append('  ── typing %d ──' % (ti + 1))
                for lab, kk, legs in tm['sources']:
                    L.append('     source %-4s → %s ⟨%s⟩' % (lab, kk, legs))
                for lab, co, legs in tm['internals']:
                    L.append('     vertex %-4s → coeff %s   %s' % (lab, co, legs))
                for num, pr in tm['propagators']:
                    L.append('     propagator %-7s → %s' % (num, pr))
                exts = ', '.join('%s=%s' % (n, f) for n, f in tm['externals'])
                L.append('     external legs → %s' % exts)
            if nt > max_typings:
                L.append('  … +%d more typing(s) (in the returned data)'
                         % (nt - max_typings))
    return '\n'.join(L)


if __name__ == '__main__':
    plot_prediagrams('ou_quartic', 2, 1, save='/tmp/prediagrams_ou_quartic_k2_l1.png')

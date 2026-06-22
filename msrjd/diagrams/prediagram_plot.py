"""Contributing-prediagram plots, grouped by topology family.

For a theory at correlator order ``k`` and loop order ``max_ell``, draw the
MSR-JD *prediagrams* (directed topologies) that survive the vertex/source
filter -- i.e. the ones the theory can actually realise -- in the Buice/Ocker
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
from matplotlib.patches import FancyArrowPatch, Circle

XS, YS = 2.9, 1.9   # layer (x) and within-layer (y) spacing


def _roman(n):
    out, vals = '', [(10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')]
    for v, s in vals:
        while n >= v: out += s; n -= v
    return out


def contributing_prediagrams(model, k, max_ell):
    """The prediagrams (filtered directed topologies) contributing to the
    ``k``-point cumulant up to loop order ``max_ell`` for ``model`` (a model
    dict from :func:`daedalus.load_theory`).  Returns ``{ell: [prediagram]}``
    where each prediagram is the ``(D, G, leaves, internal)`` tuple from the
    enumerator (``D`` a Sage DiGraph)."""
    from msrjd.core.field_theory import FieldTheory
    from msrjd.core.vertices import extract_vertex_types, extract_source_types
    from msrjd.enumeration.loop_diagram_enumeration import enumerate_all as enum_pre
    from msrjd.diagrams.filter import filter_prediagrams
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


def _roles(D, leaves):
    from msrjd.diagrams.filter import classify_prediagram_vertices
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
    g = pydot.Dot(graph_type='digraph', rankdir='RL', nodesep='0.5', ranksep='0.95')
    for n in nodes:
        g.add_node(pydot.Node(str(n), shape='point', width='0.12'))
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
    for v in pos: pos[v] = [pos[v][0] * sc, pos[v][1] * sc]
    return pos, ext_v, src_v, int_v, edges


def _layout(D, leaves):
    try:
        return _layout_graphviz(D, leaves)
    except Exception:
        return _layout_layered(D, leaves)


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
            sp = YS if endcap else YS * 0.55
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
    maxmult = max(mult.values()) if mult else 1
    idegs = tuple(sorted((int(D.in_degree(v)), int(D.out_degree(v))) for v in int_v))
    return (len(int_v), len(src_v), idegs, maxmult)


def sig_label(sig):
    n_int, n_src, idegs, maxmult = sig
    motif = {1: '', 2: ' · bubble', 3: ' · sunset', 4: ' · 4-edge'}.get(maxmult, ' · %d-edge' % maxmult)
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


def draw_prediagram(D, leaves, ax, title=None):
    pos, ext_v, src_v, int_v, edges, srclab, intlab, extlab, ext_order, _ = _generic_labels(D, leaves)
    BLACK, EDGEC = '#181818', '#3a3a3a'
    # Rank index per x-column -- scale-INDEPENDENT, so a "skip" edge is detected
    # as long whatever the layout's absolute units (graphviz vs the fallback).
    xcols = sorted(set(round(pos[v][0], 2) for v in pos))
    rank_of = {x: i for i, x in enumerate(xcols)}
    def _span(u, w):
        return abs(rank_of[round(pos[u][0], 2)] - rank_of[round(pos[w][0], 2)])
    pair = collections.defaultdict(list)
    for e in edges: pair[frozenset((e[0], e[1]))].append(e)
    track = [list(pos[v]) for v in pos]              # extent: nodes + arc apexes + numbers
    for key, es in pair.items():
        m = len(es)
        for i, e in enumerate(es):
            u, v, _ = e
            if m > 1:                                    # parallel edges -> lens bubble
                rad = 0.42 * (i - (m - 1) / 2.0) * 2
            else:
                # Keep the edge STRAIGHT unless its segment passes too close to an
                # intermediate node; only then arc it (and bow away from that node).
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ddx, ddy = x1 - x0, y1 - y0
                seg_L2 = ddx * ddx + ddy * ddy
                ru, rv = rank_of[round(x0, 2)], rank_of[round(x1, 2)]
                blockers = []
                for w in pos:
                    if w == u or w == v:
                        continue
                    xw, yw = pos[w]
                    rw = rank_of[round(xw, 2)]
                    if not (min(ru, rv) < rw < max(ru, rv)):
                        continue                          # only strictly-intermediate ranks block
                    t = 0.5 if seg_L2 < 1e-12 else max(0.0, min(
                        1.0, ((xw - x0) * ddx + (yw - y0) * ddy) / seg_L2))
                    if ((xw - (x0 + t * ddx)) ** 2 + (yw - (y0 + t * ddy)) ** 2) ** 0.5 < 0.32:
                        blockers.append(w)
                if not blockers:
                    rad = 0.0                              # nothing in the way -> straight
                else:
                    soff = 0.0                             # which side are the blockers on?
                    for w in blockers:
                        t = 0.5 if abs(ddx) < 1e-9 else (pos[w][0] - x0) / ddx
                        soff += pos[w][1] - (y0 + t * ddy)
                    side = -1.0 if soff >= 0 else 1.0      # bow AWAY from the blockers
                    span = max(_span(u, v), 1)
                    # matplotlib arc3 apex_y ~ −rad·dx, so rad = −side·sign(dx)·mag
                    rad = -min(0.72 / span, 0.45) * side * (1 if ddx >= 0 else -1)
            dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
            mx, my = (pos[u][0] + pos[v][0]) / 2.0, (pos[u][1] + pos[v][1]) / 2.0
            cx, cy = mx + rad * dy, my - rad * dx          # arc3 control point
            P0, P1 = pos[u], pos[v]
            def _bez(t, P0=P0, P1=P1, cx=cx, cy=cy):       # point on the (curved) edge
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
            track.append(list(_bez(0.5)))                  # arc apex (for the limits)
    # Propagators carry no symbol of their own -- they are named by their endpoints
    # (a→b) in the mapping table, so the figure stays uncluttered.
    for v in ext_v:                                        # external legs: hollow, labelled 1,2,…
        ax.add_patch(Circle(pos[v], 0.16, fc='white', ec=BLACK, lw=1.8, zorder=3))
        ax.text(pos[v][0] - 0.34, pos[v][1], extlab[v], ha='right', va='center', fontsize=10.5)
        track.append([pos[v][0] - 0.55, pos[v][1]])        # room for the leg label
    for v in int_v:
        ax.add_patch(Circle(pos[v], 0.17, fc=BLACK, ec=BLACK, zorder=3))
        ax.text(pos[v][0], pos[v][1] + 0.36, intlab[v], ha='center', va='bottom', fontsize=11, fontweight='bold')
    for v in src_v:
        ax.add_patch(Circle(pos[v], 0.17, fc=BLACK, ec=BLACK, zorder=3))
        ax.text(pos[v][0] + 0.34, pos[v][1], srclab[v], ha='left', va='center', fontsize=11, fontstyle='italic')
    # Limits enclose nodes, arc apexes and the leg labels, so nothing is clipped;
    # a little extra room on the right/top for the source/internal labels.
    xs = [p[0] for p in track]; ys = [p[1] for p in track]
    ax.set_xlim(min(xs) - 0.4, max(xs) + 0.95); ax.set_ylim(min(ys) - 0.55, max(ys) + 0.6)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=10, pad=2)


# ── grouped figure (public entry point) ───────────────────────────────
def plot_prediagrams(model, k, max_ell, save=None, ncol=3):
    """Draw the contributing prediagrams for ``model`` at correlator order
    ``k`` and loop order ``max_ell``, grouped by topology family and labelled
    by role (sources i,ii right; internals a,b,c middle; external legs 1,2 left;
    propagators named by endpoints, e.g. a→b).  ``model`` is a model dict (from
    :func:`daedalus.load_theory`) or a theory-name string.  Returns the
    matplotlib ``Figure``; also writes a PNG when ``save`` is given."""
    from matplotlib.gridspec import GridSpec
    if isinstance(model, str):
        import daedalus as dd
        name = model; model, _ = dd.load_theory(model)
    else:
        name = model.get('name', 'model')
    pre = contributing_prediagrams(model, k, max_ell)
    groups = []   # (ell, sig, [prediagrams]) ordered
    for ell in sorted(pre):
        bysig = collections.OrderedDict()
        for pd in pre[ell]:
            bysig.setdefault(topo_signature(pd[0], pd[2]), []).append(pd)
        for sig in sorted(bysig, key=lambda s: (s[0], s[1], -s[3])):
            groups.append((ell, sig, bysig[sig]))
    # build a row plan: a thin header row per group, then diagram rows
    HEAD, CELL = 0.5, 3.5
    plan = []   # ('head', text) | ('row', [pds])
    for (ell, sig, pds) in groups:
        plan.append(('head', r'$\ell=%d$   ·   %s   ·   %d diagram%s'
                     % (ell, sig_label(sig), len(pds), '' if len(pds) == 1 else 's')))
        for c in range(0, len(pds), ncol):
            plan.append(('row', pds[c:c + ncol]))
    if not plan:                          # nothing contributes at this (k, ell)
        fig = plt.figure(figsize=(6.5, 1.8))
        fig.text(0.5, 0.5, 'No contributing prediagrams for %s at k=%d, ℓ≤%d'
                 % (name, k, max_ell), ha='center', va='center', fontsize=12)
        if save:
            fig.savefig(save, dpi=145, bbox_inches='tight')
        return fig
    heights = [HEAD if p[0] == 'head' else CELL for p in plan]
    fig = plt.figure(figsize=(4.1 * ncol, sum(heights) + 0.8))
    gs = GridSpec(len(plan), ncol, height_ratios=heights, hspace=0.45, wspace=0.20,
                  left=0.02, right=0.98, top=1 - 0.5 / (sum(heights) + 0.8), bottom=0.4 / (sum(heights) + 0.8))
    for ri, p in enumerate(plan):
        if p[0] == 'head':
            ax = fig.add_subplot(gs[ri, :]); ax.axis('off')
            ax.add_patch(plt.Rectangle((0, 0.12), 1, 0.76, transform=ax.transAxes, fc='#eef2f8', ec='none', zorder=0))
            ax.text(0.012, 0.5, p[1], transform=ax.transAxes, fontsize=11, fontweight='bold',
                    color='#1c3a66', va='center', ha='left')
        else:
            for ci, pd in enumerate(p[1]):
                ax = fig.add_subplot(gs[ri, ci])
                draw_prediagram(pd[0], pd[2], ax, title='#%d' % (ci + 1))
    fig.suptitle('Contributing prediagrams: %s,  k=%d, ℓ≤%d' % (name, k, max_ell),
                 fontsize=13, y=0.998)
    fig.text(0.5, 0.008, r'time $\leftarrow$      $\circ\;1,2$ external legs      '
             r'$\bullet\;i,ii$ sources      $\bullet\;a,b,c$ internal vertices      '
             r'(propagators named by endpoints, e.g. $a\to b$)',
             ha='center', fontsize=10, color='#555')
    if save:
        fig.savefig(save, dpi=145, bbox_inches='tight')
    return fig


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
        name = model; model, _ = dd.load_theory(model)
    else:
        name = model.get('name', 'model')
    from msrjd.core.field_theory import FieldTheory
    from msrjd.core.vertices import extract_vertex_types, extract_source_types
    from msrjd.diagrams.type_assignment import (enumerate_typed_diagrams,
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
        from pipeline._propagator import build_propagator
        G_ft = build_propagator(ft, model, verbose=False)['G_ft']
    pre = contributing_prediagrams(model, k, max_ell)
    result = {}
    for ell in sorted(pre):
        bysig = collections.OrderedDict()
        for pd in pre[ell]:
            bysig.setdefault(topo_signature(pd[0], pd[2]), []).append(pd)
        entries = []
        for sig in sorted(bysig, key=lambda s: (s[0], s[1], -s[3])):
            for idx, pd in enumerate(bysig[sig]):
                labels = _generic_labels(pd[0], pd[2])
                typings = [diagram_label_map(td, labels)
                           for td in enumerate_typed_diagrams(
                               pd, external_fields, vt, st, G_ft, ri, pi)]
                entries.append({'sig': sig, 'idx': idx + 1,
                                'labels': labels, 'typings': typings})
        result[ell] = entries
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
    make_figure('ou_quartic', 2, 1, '/tmp/v3_ou_quartic_k2_l1.png')

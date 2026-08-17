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

from collections import defaultdict

__all__ = ['causal_depths', 'layout_typed_diagram', 'layout_prediagram',
           'DX', 'DY', 'DY_EXTERNAL']

# Column pitch (x) and row pitch (y) in TikZ units.
DX = 2.3
DY = 1.25
# External vertices sit BESIDE their label rather than inside it, so the node
# is a small circle and needs only slightly more room than an internal dot.
DY_EXTERNAL = 1.5


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


def layout_typed_diagram(diagram, dx=DX, dy=DY, order_externals=True):
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
    _order_layers(layer_of, layers, adj)
    if not order_externals:
        layers[ext_depth] = fixed_ext

    dy_ext = dy * (DY_EXTERNAL / DY)

    def pitch_of(d):
        return dy_ext if d == ext_depth else dy

    y_of = {}
    for d, col in layers.items():
        pitch = pitch_of(d)
        n = len(col)
        for i, v in enumerate(col):
            y_of[v] = (i - (n - 1) / 2.0) * pitch
    _relax_y(layers, adj, y_of, pitch_of)

    if y_of:                                   # re-centre about y = 0
        mid = 0.5 * (max(y_of.values()) + min(y_of.values()))
        for v in y_of:
            y_of[v] -= mid

    return {v: (-layer_of[v] * dx, y_of[v]) for v in D.vertices()}


layout_prediagram = layout_typed_diagram

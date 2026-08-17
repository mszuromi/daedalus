"""Geometry for drawing a typed diagram.

Pure layout: this module assigns an ``(x, y)`` to every vertex and never
emits markup.  Keeping it separate means the placement rules can be tested
on coordinates alone, without parsing TikZ.

Convention (matches the figures in the paper)
---------------------------------------------
**Time flows right to left.**  A prediagram's edges are oriented along the
retarded propagator, earlier -> later, so a vertex's *causal depth* (the
longest directed path reaching it from any source) increases with time.  We
therefore map larger depth to smaller ``x``:

    sources           depth 0            rightmost
    interactions      depth 1, 2, ...    middle columns
    external legs     pinned             flush left, aligned

External legs are pinned to a common ``x`` rather than placed by their own
depth, so every diagram in a figure array has its observation points on the
same vertical line.  Within a column, vertices are spread symmetrically
about ``y = 0``.
"""

from collections import defaultdict

__all__ = ['causal_depths', 'layout_typed_diagram', 'layout_prediagram',
           'DX', 'DY', 'DY_EXTERNAL']

# Column pitch (x) and row pitch (y) in TikZ units.  Sized so a vertex-factor
# label hanging below a vertex clears the ``G`` labels on the propagators
# converging there -- at tighter pitches the two collide and the figure is
# unreadable, which is the first thing that goes wrong when this is tuned down.
DX = 2.3
DY = 1.5
# External vertices are drawn as circles around an inline label
# (``\delta x(y_4)``), so their nodes are far wider and taller than the bare
# dots used for sources and interactions.  At k>=4 a shared pitch makes the
# external circles touch; give that column its own, larger spacing.
DY_EXTERNAL = 2.1


def _unpack(diagram):
    """Return ``(D, leaves)`` from a TypedDiagram or a raw prediagram tuple."""
    prediagram = getattr(diagram, 'prediagram', diagram)
    D, _G, leaves, _internal = prediagram
    return D, list(leaves)


def causal_depths(D):
    """Longest-path depth of every vertex from the set of sources.

    Sources are the in-degree-0 vertices (all-response, no incoming
    propagator).  The orientation constraints guarantee the digraph is
    acyclic, so a relaxation over a topological order is exact.

    Returns ``{vertex: depth}``.
    """
    depth = {v: 0 for v in D.vertices()}
    try:
        order = D.topological_sort()
    except Exception:
        # Should not happen (orientations are constrained to DAGs), but a
        # drawing routine must never be the thing that raises.
        order = list(D.vertices())
    for u in order:
        for v in D.neighbors_out(u):
            if depth[u] + 1 > depth[v]:
                depth[v] = depth[u] + 1
    return depth


def _spread(n, pitch=DY):
    """``n`` positions centred on zero: (-p, 0, p) for n=3, (-p/2, p/2) for n=2."""
    if n <= 1:
        return [0.0]
    return [(i - (n - 1) / 2.0) * pitch for i in range(n)]


def layout_typed_diagram(diagram, dx=DX, dy=DY):
    """Assign ``(x, y)`` to every vertex of a typed diagram or prediagram.

    Parameters
    ----------
    diagram : TypedDiagram or (D, G, leaves, internal)
    dx, dy : float
        Column and row pitch.

    Returns
    -------
    dict
        ``{vertex: (x, y)}``.  External (leaf) vertices share the smallest
        ``x``; sources sit at ``x = 0``.
    """
    D, leaves = _unpack(diagram)
    leaf_set = set(leaves)
    depth = causal_depths(D)

    internal = [v for v in D.vertices() if v not in leaf_set]
    max_internal_depth = max((depth[v] for v in internal), default=0)

    pos = {}

    # Internal vertices: one column per causal depth, right to left.
    by_depth = defaultdict(list)
    for v in internal:
        by_depth[depth[v]].append(v)
    for d, verts in by_depth.items():
        verts.sort()
        for v, y in zip(verts, _spread(len(verts), dy)):
            pos[v] = (-d * dx, y)

    # External legs: pinned one column left of the deepest internal vertex,
    # ordered by leaf index so the k external points read top-to-bottom in a
    # stable order across diagrams.
    x_ext = -(max_internal_depth + 1) * dx
    dy_ext = dy * (DY_EXTERNAL / DY)
    for v, y in zip(sorted(leaves), _spread(len(leaves), dy_ext)):
        pos[v] = (x_ext, y)

    return pos


# Backwards-compatible alias: the layout is identical for an untyped
# prediagram, which carries the same (D, G, leaves, internal) payload.
layout_prediagram = layout_typed_diagram

"""Layout geometry and tikz-feynman emission.

The emitter has no numerical content, so these are structural checks: that
time runs right to left, that parallel edges are bowed apart rather than
drawn on top of each other, and that the paper's vertex conventions survive
into the output.
"""

import pytest

import engine.enumeration.loop_diagram_enumeration as L
from engine.diagrams.typed_diagram_layout import (
    causal_depths, layout_typed_diagram, legibility_penalty,
    vertex_edge_clearances, fan_separations, mm_per_unit,
    edge_edge_clearances, crossing_separations,
    DOT_RADIUS_MM, ARROWHEAD_MM,
)
from engine.diagrams.tikz_export import (
    to_tikz_feynman, diagram_to_standalone, DEFAULT_SYMBOL_MAP,
    edge_bends, path_point, arrow_positions, place_labels, MIN_BUBBLE_MM,
    _LABEL_MARGIN_MM, _INNER_SEP_MM, _PT_MM,
    _label_box, _label_extent_mm, _box_overlap, _seg_in_box,
)


@pytest.fixture(scope='module')
def prediagrams_2_1():
    return L.enumerate_all(2, 1, verbose=False)[2]


@pytest.fixture(scope='module')
def prediagrams_2_2():
    """Two loops -- where the layout actually gets hard (283 diagrams)."""
    return L.enumerate_all(2, 2, verbose=False)[2]


# ── layout ──────────────────────────────────────────────────────────

def test_sources_have_depth_zero(prediagrams_2_1):
    for D, _G, _lv, _in in prediagrams_2_1:
        depth = causal_depths(D)
        for v in D.vertices():
            if D.in_degree(v) == 0:
                assert depth[v] == 0, 'a source must sit at depth 0'


def test_time_flows_right_to_left(prediagrams_2_1):
    """Every propagator must point leftward: head strictly left of tail."""
    for D, _G, leaves, _in in prediagrams_2_1:
        pos = layout_typed_diagram((D, _G, leaves, _in))
        for u, v in D.edges(labels=False):
            assert pos[v][0] < pos[u][0] + 1e-9, (
                f'edge {u}->{v} does not advance leftward in time')


def test_externals_are_flush_left(prediagrams_2_1):
    for D, _G, leaves, internal in prediagrams_2_1:
        pos = layout_typed_diagram((D, _G, leaves, internal))
        xs_ext = {round(pos[v][0], 9) for v in leaves}
        assert len(xs_ext) == 1, 'external legs must share one x'
        x_ext = xs_ext.pop()
        for v in internal:
            assert pos[v][0] > x_ext - 1e-9, 'no internal vertex left of the externals'


def test_layout_covers_every_vertex(prediagrams_2_1):
    for D, _G, leaves, internal in prediagrams_2_1:
        pos = layout_typed_diagram((D, _G, leaves, internal))
        assert set(pos) == set(D.vertices())


# ── emission ────────────────────────────────────────────────────────

def test_emits_one_node_per_vertex_and_one_edge_per_propagator(prediagrams_2_1):
    for pd in prediagrams_2_1:
        D = pd[0]
        tex = to_tikz_feynman(pd)
        assert tex.count(r'\vertex') == D.order()
        assert tex.count(' -- [') == D.size()


def test_external_and_internal_vertex_styles(prediagrams_2_1):
    D, _G, leaves, internal = prediagrams_2_1[0]
    tex = to_tikz_feynman((D, _G, leaves, internal))
    assert tex.count('empty dot') == len(leaves), 'externals unshaded'
    # '[dot,' or '[dot]' -- the option list may carry a factor label
    assert (tex.count('[dot,') + tex.count('[dot]')) == len(internal), \
        'sources/interactions solid'
    # spacing after \delta is required for single-letter fields (\delta x,
    # not \deltax), so match the parts rather than an exact string
    assert r'\delta' in tex and r'(y_{1})' in tex


def test_parallel_edges_are_bowed_apart(prediagrams_2_1):
    """A loop between two vertices must not draw two coincident lines."""
    checked = False
    for pd in prediagrams_2_1:
        D = pd[0]
        counts = {}
        for e in D.edges(labels=False):
            counts[e] = counts.get(e, 0) + 1
        if not any(c > 1 for c in counts.values()):
            continue
        checked = True
        tex = to_tikz_feynman(pd)
        assert 'bend left=' in tex and 'bend right=' in tex, (
            'parallel propagators must be bent in opposite directions')
        # a semicircular ``half left/right`` balloons between distant
        # vertices; the bow must be a bounded angle instead
        assert 'half left' not in tex and 'half right' not in tex
    assert checked, 'expected at least one multi-edge prediagram at (2,1)'


def test_propagator_label_is_configurable(prediagrams_2_1):
    pd = prediagrams_2_1[0]
    # ``edge label'`` (primed) puts the label on the opposite side of the
    # line from the vertex factors, which ride above their vertices.
    assert r"edge label'=\(G\)" in to_tikz_feynman(pd)
    assert r"edge label'=\(R\)" in to_tikz_feynman(pd, propagator_label='R')
    assert 'edge label' not in to_tikz_feynman(pd, propagator_label=None)


def test_standalone_document_is_self_contained(prediagrams_2_1):
    doc = diagram_to_standalone(prediagrams_2_1[0])
    assert doc.startswith(r'\documentclass')
    assert 'tikz-feynman' in doc
    assert doc.rstrip().endswith(r'\end{document}')
    assert doc.count(r'\begin{tikzpicture}') == 1


def test_untyped_prediagram_draws_without_a_model(prediagrams_2_1):
    """No vertex_assignments -> no factor labels, but still a valid picture."""
    tex = to_tikz_feynman(prediagrams_2_1[0])
    assert r'\begin{feynman}' in tex and r'\end{feynman}' in tex


def test_symbol_map_rewrites_parameter_names():
    class _Coeff:
        def _latex_(self):
            return r'3 \, \mathit{eps} \mathit{xstar}_{1}'

    class _V:
        coefficient = _Coeff()

    class _TD:
        prediagram = None
        vertex_assignments = {}

    from engine.diagrams.tikz_export import _vertex_label
    out = _vertex_label(_V(), True, DEFAULT_SYMBOL_MAP)
    assert r'\varepsilon' in out
    # the saddle renders in the model's OWN field name, matching the
    # external labels, rather than a hard-coded \phi
    assert r'x^{*}' in out
    assert 'mathit' not in out


def test_saddle_symbols_render_for_unmapped_fields():
    """<field>star must work for a model the explicit map never listed."""
    from engine.diagrams.tikz_export import _apply_symbol_map, DEFAULT_SYMBOL_MAP
    out = _apply_symbol_map(r'3 \, \varepsilon_{2} \mathit{ystar}_{1}',
                            DEFAULT_SYMBOL_MAP)
    assert r'y^{*}' in out and 'mathit' not in out
    assert r'n^{*}' in _apply_symbol_map(r'\mathit{nstar}_{2}',
                                         DEFAULT_SYMBOL_MAP)


# ── compilation ─────────────────────────────────────────────────────

@pytest.mark.slow
def test_standalone_document_compiles(tmp_path, prediagrams_2_1):
    """The emitted document must actually build under pdflatex.

    Skipped where no TeX toolchain or no tikz-feynman is installed -- this is
    the only check that catches a syntactically valid but semantically broken
    picture (e.g. an option tikz-feynman does not accept).
    """
    import shutil, subprocess
    if shutil.which('pdflatex') is None:
        pytest.skip('pdflatex not installed')
    src = tmp_path / 'd.tex'
    src.write_text(diagram_to_standalone(prediagrams_2_1[0]))
    proc = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'd.tex'],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 and 'tikz-feynman' in (proc.stdout + proc.stderr):
        pytest.skip('tikz-feynman package not installed')
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert (tmp_path / 'd.pdf').exists()


# ── multi-field labelling ───────────────────────────────────────────

def test_field_symbol_strips_response_and_delta_markers():
    from engine.diagrams.tikz_export import _field_symbol
    assert _field_symbol(('yt', 1)) == 'y'    # response: trailing t
    assert _field_symbol(('dx', 1)) == 'x'    # physical: leading d
    assert _field_symbol(('n', 0)) == 'n'     # unmarked passes through
    assert _field_symbol(None) == ''


def test_auto_propagator_label_names_the_matrix_element():
    """In a multi-field model different edges are different G_{ab}."""
    from engine.diagrams.tikz_export import _edge_label

    class _TD:
        edge_types = {(2, 1, 0): (('yt', 1), ('dx', 1)),
                      (3, 2, 0): (('xt', 1), ('dx', 1))}

    td = _TD()
    assert _edge_label(td, (2, 1, 0), 'auto') == 'G_{yx}'
    # same field on both ends -> single subscript, not "G_{xx}"
    assert _edge_label(td, (3, 2, 0), 'auto') == 'G_{x}'
    # unknown edge falls back rather than raising
    assert _edge_label(td, (9, 9, 9), 'auto') == 'G'
    # an explicit label always wins
    assert _edge_label(td, (2, 1, 0), 'G') == 'G'


def test_auto_external_label_uses_the_field(prediagrams_2_1):
    """Externals of different fields must not all read \delta\phi."""
    D, _G, leaves, internal = prediagrams_2_1[0]

    class _TD:
        prediagram = (D, _G, leaves, internal)
        vertex_assignments = {}
        external_legs = {leaves[0]: ('dx', 1), leaves[1]: ('dy', 1)}

    tex = to_tikz_feynman(_TD())
    assert r'\delta x(y_{1})' in tex
    assert r'\delta y(y_{2})' in tex


def test_no_self_loops_to_draw():
    """Self-loops are structurally impossible, so the emitter need not draw them.

    ``add_edges_to_tree`` builds with ``loops=False`` and draws candidate edges
    from ``combinations(V, 2)``, so a vertex can never connect to itself.  That
    matches the physics: the retarded propagator has G(t,t)=0 under Ito, so a
    tadpole would vanish anyway.  Pinned here because a drawing routine that
    silently mishandled one would be hard to notice.
    """
    import engine.enumeration.loop_diagram_enumeration as L
    for k, ell in ((2, 1), (2, 2), (3, 1)):
        for D, _G, _lv, _in in L.enumerate_all(k, ell, verbose=False)[2]:
            assert not any(u == v for u, v in D.edges(labels=False))


def test_external_column_is_spaced_wider_than_internal_rows():
    """External nodes carry inline text, so they need more room than dots."""
    import engine.enumeration.loop_diagram_enumeration as L
    from engine.diagrams.typed_diagram_layout import DY, DY_EXTERNAL
    assert DY_EXTERNAL > DY
    pd = [p for p in L.enumerate_all(4, 0, verbose=False)[2]][0]
    pos = layout_typed_diagram(pd)
    leaves = sorted(pd[2])
    ys = sorted(pos[v][1] for v in leaves)
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    assert all(g >= DY_EXTERNAL - 1e-9 for g in gaps), (
        f'external rows {gaps} tighter than the external pitch {DY_EXTERNAL}')


def test_labels_ride_outside_their_nodes(prediagrams_2_1):
    """Neither an external name nor a vertex factor may sit in the node BODY.

    An ``[empty dot]`` sized to contain ``\delta x(y_1)`` becomes a huge
    circle with text inside it, and a filled ``[dot]`` draws its body text
    inside the ink where it is invisible.  Both must use ``label=``.
    """
    D, _G, leaves, internal = prediagrams_2_1[0]
    tex = to_tikz_feynman((D, _G, leaves, internal))
    for line in tex.splitlines():
        if r'\vertex' not in line:
            continue
        body = line.rsplit(')', 1)[-1]        # what follows the coordinate
        assert '{}' in body or body.strip() in (';', ''), (
            f'node body must be empty, got: {line.strip()}')
    assert 'label=' in tex, 'labels must be carried as options'


def test_external_labels_sit_to_the_left(prediagrams_2_1):
    """180 degrees keeps the name clear of the diagram it annotates."""
    D, _G, leaves, internal = prediagrams_2_1[0]
    tex = to_tikz_feynman((D, _G, leaves, internal))
    for line in tex.splitlines():
        if 'empty dot' in line:
            assert '180:' in line, f'external label not at 180: {line.strip()}'


def test_layout_minimises_drawn_crossings(prediagrams_2_1):
    """Every diagram must be laid out at its minimum crossing count.

    Pinned because the first three attempts each got this wrong in a
    different way: barycentre alone stalled at a local minimum; adjacent-swap
    refinement could not escape it either; and a dummy-vertex layer count
    optimised a ROUTE the renderer discards, since a multi-layer edge is drawn
    as one straight line.  Scoring the drawn geometry is what actually works.
    """
    import itertools, math
    from collections import defaultdict
    from engine.diagrams.typed_diagram_layout import (
        causal_depths, _neighbours, _relax_y, _geometric_crossings,
        DX, DY, DY_EXTERNAL,
    )
    for pd in prediagrams_2_1:
        D, _G, leaves, _internal = pd
        leaf = set(leaves)
        dep = causal_depths(D)
        adj = _neighbours(D)
        ext = max((dep[v] for v in D.vertices() if v not in leaf),
                  default=0) + 1
        lo = {v: (ext if v in leaf else dep[v]) for v in D.vertices()}
        edges = list(D.edges(labels=False))

        drawn = _geometric_crossings(layout_typed_diagram(pd), edges)

        layers = defaultdict(list)
        for v in sorted(D.vertices()):
            layers[lo[v]].append(v)
        depths = sorted(layers)
        if math.prod(math.factorial(len(layers[d])) for d in depths) > 5000:
            continue                       # exhaustive check too costly here
        dy_ext = DY * (DY_EXTERNAL / DY)

        def pitch(d, _e=ext, _de=dy_ext):
            return _de if d == _e else DY

        best = None
        for combo in itertools.product(
                *(itertools.permutations(layers[d]) for d in depths)):
            cand = {d: list(c) for d, c in zip(depths, combo)}
            y = {}
            for d, col in cand.items():
                n = len(col)
                for j, v in enumerate(col):
                    y[v] = (j - (n - 1) / 2.0) * pitch(d)
            _relax_y(cand, adj, y, pitch)
            c = _geometric_crossings(
                {v: (-lo[v] * DX, y[v]) for v in y}, edges)
            if best is None or c < best:
                best = c
            if best == 0:
                break
        assert drawn == best, (
            f'laid out with {drawn} crossings, minimum is {best}')


# ── legibility: a drawing must not read as a different graph ────────

def test_penalty_sees_a_vertex_sitting_on_a_propagator():
    """The unit check behind the two set-wide tests below.

    Zero crossings is not the same as legible: this configuration has none
    and is still unreadable, because vertex 2 lies exactly on the propagator
    1->3 and the segment 1->2 is drawn on top of the first half of it.
    """
    pos = {1: (0.0, 0.0), 2: (-2.3, 0.0), 3: (-4.6, 0.0)}
    edges = [(1, 3), (1, 2)]
    worst = min(d for d, _v, _e in vertex_edge_clearances(pos, edges))
    assert worst == pytest.approx(0.0, abs=1e-9)
    assert min(d for d, _a, _b in fan_separations(pos, edges)) \
        == pytest.approx(0.0, abs=1e-9)
    assert legibility_penalty(pos, edges) > 0.0

    pos[2] = (-2.3, 1.0)                      # lift it off the line
    assert min(d for d, _v, _e in vertex_edge_clearances(pos, edges)) \
        > 2 * DOT_RADIUS_MM
    assert legibility_penalty(pos, edges) == 0.0


def test_printed_scale_shrinks_with_width():
    """A wide diagram is scaled down to fit its panel, so its ink is smaller.

    The legibility thresholds are in printed millimetres, which is only
    meaningful because this conversion tracks the panel shrink.
    """
    narrow = {1: (0.0, 0.0), 2: (-2.3, 1.0)}
    wide = {1: (0.0, 0.0), 2: (-23.0, 1.0)}
    assert mm_per_unit(narrow) > mm_per_unit(wide)
    assert mm_per_unit(wide) * 23.0 == pytest.approx(46.0)


def test_no_vertex_is_drawn_on_an_unrelated_propagator(prediagrams_2_2):
    """No propagator may pass through a vertex dot it is not attached to.

    Measured, not eyeballed: the drawn straight segments against the real
    tikz-feynman dot, whose radius was taken off a 400 dpi render.  Before
    the placement objective scored this, 11 pairs across 10 of the 66 printed
    two-loop panels were closer than the dot RADIUS -- the worst at 0.03 mm,
    i.e. the line running through the middle of the ink, which made the panel
    read as a chain of two propagators instead of one passing behind a
    vertex.
    """
    for pd in prediagrams_2_2:
        edges = list(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        d, v, e = min(vertex_edge_clearances(pos, edges), default=(9e9, 0, 0))
        assert d >= 2 * DOT_RADIUS_MM, (
            f'vertex {v} is {d:.2f} mm from propagator {e} '
            f'(dot radius {DOT_RADIUS_MM} mm)')


def test_edges_leaving_a_vertex_stay_apart(prediagrams_2_2):
    """Two propagators out of one vertex must be told apart where they part.

    The separation is taken at the nearer far endpoint, which is where a
    near-parallel pair is still superimposed.  Below one arrowhead width the
    pair prints as a single line with a hairline sliver; that fired 28 times
    on 18 of the 66 two-loop panels before the fix.
    """
    for pd in prediagrams_2_2:
        edges = list(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        d, a, b = min(fan_separations(pos, edges), default=(9e9, 0, 0))
        assert d >= ARROWHEAD_MM, (
            f'edges {a} and {b} are only {d:.2f} mm apart where they '
            f'separate (arrowhead is {ARROWHEAD_MM} mm wide)')


def test_penalty_sees_two_unrelated_propagators_laid_together():
    """Zero crossings, every vertex clear, every fan wide -- and unreadable.

    Edges 1-2 and 3-4 share no vertex, never cross, and no vertex is near
    either of them, so the crossing count and both of the older legibility
    terms score this drawing as perfect.  It still prints as ONE propagator,
    because the two lie 0.1 mm apart along their whole length.  This is the
    blind spot the task named as defect 2.
    """
    pos = {1: (0.0, 0.0), 2: (-4.6, 0.0), 3: (0.0, 0.01), 4: (-4.6, 0.01)}
    edges = [(1, 2), (3, 4)]
    assert min(d for d, _v, _e in vertex_edge_clearances(pos, edges)) \
        == pytest.approx(0.1, abs=0.05)
    assert list(fan_separations(pos, edges)) == []
    gap = min(d for d, _a, _b in edge_edge_clearances(pos, edges))
    assert gap < ARROWHEAD_MM
    assert legibility_penalty(pos, edges) > 0.0

    pos[3], pos[4] = (0.0, 1.0), (-4.6, 1.0)      # pull them apart
    assert min(d for d, _a, _b in edge_edge_clearances(pos, edges)) \
        > ARROWHEAD_MM
    assert legibility_penalty(pos, edges) == 0.0


def test_penalty_sees_a_glancing_crossing():
    """A crossing is fine at a right angle and a misread at a glancing one.

    Same graph, same crossing count, same everything the older terms
    measure; only the angle differs.  At 3 degrees the two strands run
    merged either side of the crossing and the reader cannot tell which
    continues into which, so the drawing reads as a junction the graph does
    not have.
    """
    steep = {1: (0.0, -2.0), 2: (-4.6, 2.0), 3: (0.0, 2.0), 4: (-4.6, -2.0)}
    edges = [(1, 2), (3, 4)]
    assert min(d for d, _a, _b in crossing_separations(steep, edges)) \
        > ARROWHEAD_MM
    assert legibility_penalty(steep, edges) == 0.0

    glancing = {1: (0.0, 0.0), 2: (-4.6, 0.12),
                3: (0.0, 0.12), 4: (-4.6, 0.0)}
    assert min(d for d, _a, _b in crossing_separations(glancing, edges)) \
        < ARROWHEAD_MM
    assert legibility_penalty(glancing, edges) > 0.0


def test_unrelated_propagators_stay_apart(prediagrams_2_2):
    """No two propagators may be drawn within an arrowhead of each other.

    The set-wide form of the unit check above.  An arrowhead is the widest
    ink on a propagator, so a pair closer than that cannot both carry a
    legible mark and `arrow_positions` has nowhere along the line to move
    one to.
    """
    for pd in prediagrams_2_2:
        edges = list(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        d, a, b = min(edge_edge_clearances(pos, edges), default=(9e9, 0, 0))
        assert d >= ARROWHEAD_MM, (
            f'propagators {a} and {b} run {d:.2f} mm apart '
            f'(arrowhead is {ARROWHEAD_MM} mm wide)')


def test_crossings_are_not_glancing(prediagrams_2_2):
    """Where two propagators must cross, they must part promptly."""
    for pd in prediagrams_2_2:
        edges = list(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        d, a, b = min(crossing_separations(pos, edges), default=(9e9, 0, 0))
        assert d >= ARROWHEAD_MM, (
            f'{a} and {b} cross and are still only {d:.2f} mm apart '
            f'where the first of them ends')


def test_every_panel_finishes_legible(prediagrams_2_2):
    """The end-to-end statement: no shipped panel misreads as another graph.

    One assertion covering all four terms at once, on the whole two-loop
    set.  Before the whole-column moves and the stretch of last resort
    existed, the per-vertex repair was a coordinate descent that stalled:
    on the three-loop set 157 of 1798 panels came out of it still scoring a
    defect no single vertex could fix alone.
    """
    bad = []
    for i, pd in enumerate(prediagrams_2_2, 1):
        pos = layout_typed_diagram(pd)
        pen = legibility_penalty(pos, list(pd[0].edges(labels=False)))
        if pen > 0.0:
            bad.append((i, pen))
    assert not bad, f'{len(bad)} panels score a legibility defect: {bad[:5]}'


def test_legibility_never_costs_a_crossing(monkeypatch, prediagrams_2_2):
    """The legibility term is a strict tie-break, never a trade.

    Laid out twice -- once as shipped, once with the penalty forced to zero
    so only crossings are optimised -- the drawn crossing count must agree on
    every diagram.  This is the guarantee that makes the repair safe to run
    unconditionally, and it fails loudly if either guard (the lexicographic
    order in the search, or the crossing check inside the y nudge) is ever
    relaxed into a weighted sum.
    """
    import engine.diagrams.typed_diagram_layout as T
    edges_of = {i: list(pd[0].edges(labels=False))
                for i, pd in enumerate(prediagrams_2_2)}
    tidy = [T._geometric_crossings(layout_typed_diagram(pd), edges_of[i])
            for i, pd in enumerate(prediagrams_2_2)]
    monkeypatch.setattr(T, 'legibility_penalty', lambda *a, **k: 0.0)
    plain = [T._geometric_crossings(layout_typed_diagram(pd), edges_of[i])
             for i, pd in enumerate(prediagrams_2_2)]
    assert tidy == plain


def _arrow_points(pd):
    """Where every arrowhead is actually printed, in layout coordinates."""
    edges = sorted(pd[0].edges(labels=False))
    pos = layout_typed_diagram(pd)
    bends = edge_bends(edges, 14)
    ts = arrow_positions(pos, edges, bends)
    return pos, [path_point(pos[u], pos[v], b, t)
                 for (u, v), b, t in zip(edges, bends, ts)]


def test_arrowheads_do_not_stack_at_a_crossing(prediagrams_2_2):
    """Two arrowheads on one point read as a vertex that is not there.

    tikz-feynman marks the `fermion` arrow at the path midpoint, and a
    symmetric layered layout makes two crossing edges cross at BOTH their
    midpoints -- so the arrowheads coincide exactly.  That fired 34 times on
    30 of the 66 printed two-loop panels, 13 of them at 0.00 mm, each one a
    solid triangular blob sitting where the lines cross.
    """
    import math
    for pd in prediagrams_2_2:
        pos, heads = _arrow_points(pd)
        mm = mm_per_unit(pos)
        for i in range(len(heads)):
            for j in range(i + 1, len(heads)):
                d = math.hypot(heads[i][0] - heads[j][0],
                               heads[i][1] - heads[j][1]) * mm
                assert d >= ARROWHEAD_MM, (
                    f'two arrowheads {d:.2f} mm apart, arrowhead is '
                    f'{ARROWHEAD_MM} mm wide')


def test_no_arrowhead_lands_on_a_vertex(prediagrams_2_2):
    """An arrowhead touching a dot reads as an arrow ENTERING that vertex."""
    import math
    for pd in prediagrams_2_2:
        pos, heads = _arrow_points(pd)
        mm = mm_per_unit(pos)
        need = DOT_RADIUS_MM + ARROWHEAD_MM / 2.0
        for h in heads:
            for c in pos.values():
                d = math.hypot(h[0] - c[0], h[1] - c[1]) * mm
                assert d >= need, f'arrowhead {d:.2f} mm from a vertex dot'


def test_arrow_stays_at_the_midpoint_unless_it_must_move(prediagrams_2_2):
    """Minimal intervention: only a colliding arrow is displaced.

    The midpoint is the convention, so an edge keeps it unless the arrowhead
    would collide.  Pinned as a fraction because a routine that moved every
    arrow would satisfy the two tests above while making every diagram
    slightly wrong.
    """
    moved = total = 0
    for pd in prediagrams_2_2:
        edges = sorted(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        for t in arrow_positions(pos, edges, edge_bends(edges, 14)):
            total += 1
            moved += abs(t - 0.5) > 1e-9
    assert moved < 0.15 * total, (
        f'{moved} of {total} arrows displaced; the midpoint should be kept '
        f'except where arrowheads actually collide')


def test_fermion_and_with_arrow_are_never_combined(prediagrams_2_2):
    """`fermion` IS `plain` + an arrow at 0.5, and the keys do not compose.

    Asking for both marks the line TWICE -- two arrowheads on one propagator,
    which was the first attempt at this and is worse than the defect it was
    meant to fix.  An edge with a displaced arrow must be spelled
    `plain, with arrow=t`.
    """
    for pd in prediagrams_2_2[:40]:
        for ln in to_tikz_feynman(pd).splitlines():
            if ' -- [' not in ln:
                continue
            assert not ('fermion' in ln and 'with arrow' in ln), ln
            assert 'fermion' in ln or 'with arrow' in ln, (
                f'propagator drawn with no causal arrow: {ln.strip()}')


def test_a_bubble_opens_wide_enough_to_read_as_two_lines(prediagrams_2_2):
    """A parallel pair must not print as one thick line.

    The two copies are bowed apart by a fixed ANGLE, which bows them by a
    fixed FRACTION of the edge -- so a short bubble got a proportionally
    narrow lens.  The tightest in the two-loop figure opened to 1.33 mm,
    narrower than the vertex dots at either end.  The angle is now raised on
    short pairs until the printed gap clears `MIN_BUBBLE_MM`; the tolerance
    below absorbs the small-angle model, the 30 degree cap and the integer
    degrees tikz is given.
    """
    import math
    from collections import Counter
    for pd in prediagrams_2_2:
        edges = sorted(pd[0].edges(labels=False))
        pos = layout_typed_diagram(pd)
        mm = mm_per_unit(pos)
        bends = edge_bends(edges, 14, pos, mm)
        polys = [[path_point(pos[u], pos[v], b, k / 20.0) for k in range(21)]
                 for (u, v), b in zip(edges, bends)]
        for pair, n in Counter(edges).items():
            if n < 2:
                continue
            idx = [k for k, e in enumerate(edges) if e == pair]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    gap = max(min(math.hypot(p[0] - q[0], p[1] - q[1])
                                  for q in polys[idx[j]])
                              for p in polys[idx[i]]) * mm
                    assert gap >= 0.95 * MIN_BUBBLE_MM, (
                        f'bubble {pair} opens only {gap:.2f} mm')


def test_a_long_bubble_is_not_over_bowed(prediagrams_2_2):
    """Widening is for SHORT pairs; a long one keeps the modest default."""
    pos = {1: (0.0, 0.0), 2: (-4.6, 0.0)}
    edges = [(1, 2), (1, 2)]
    wide = edge_bends(edges, 14, pos, mm_per_unit(pos))
    assert [abs(b) for b in wide] == [14, 14]


# ── labels ──────────────────────────────────────────────────────────

def test_label_extent_matches_latex():
    """The box model is calibrated, not guessed.

    Reference widths come from ``\settowidth`` at 11pt on the labels this
    emitter actually produces.  The estimator is allowed to run a little
    small -- callers pad it -- but not by enough to call a collision clear.
    """
    for tex, w_mm, h_mm in ((r'v_2', 3.535, 2.234),
                            (r'\kappa_2', 3.887, 2.234),
                            (r'\delta x(y_1)', 10.604, 3.849)):
        w, h = _label_extent_mm(tex)
        assert 0.85 * w_mm <= w <= 1.15 * w_mm, (tex, w, w_mm)
        assert 0.75 * h_mm <= h <= 1.25 * h_mm, (tex, h, h_mm)


def test_a_lone_label_keeps_the_canonical_place():
    """Nothing in the way -> straight up, hard against the vertex."""
    got = place_labels({'a': (0.0, 0.0, 3.5, 2.2)}, [], [(0.0, 0.0)])
    assert got['a'] == (90, 1.0)


def test_a_label_moves_off_a_propagator():
    """A line through the default position must push the label elsewhere."""
    above = [[(-20.0, 3.0), (20.0, 3.0)]]        # a wire just overhead
    got = place_labels({'a': (0.0, 0.0, 3.5, 2.2)}, above, [(0.0, 0.0)])
    assert got['a'] != (90, 1.0)
    box = _label_box(0.0, 0.0, got['a'][0], got['a'][1], 3.5, 2.2, 0.825)
    assert _seg_in_box(above[0][0], above[0][1], box) == 0.0


def test_labels_are_placed_jointly_not_first_come_first_served():
    """Two vertices close together must not both claim the same spot.

    Greedy placement gives the first vertex the best angle and leaves the
    second to overlap it; the sweep re-places both against each other.
    """
    anchors = {'a': (0.0, 0.0, 5.0, 2.2), 'b': (2.0, 0.0, 5.0, 2.2)}
    got = place_labels(anchors, [], [(0.0, 0.0), (2.0, 0.0)])
    boxes = [_label_box(*anchors[k][:2], *got[k], *anchors[k][2:], 0.825)
             for k in anchors]
    # Not merely disjoint: two labels that just touch still read as one, so
    # the placement is scored on a box grown by the margin.
    grown = (boxes[0][0] - _LABEL_MARGIN_MM, boxes[0][1] - _LABEL_MARGIN_MM,
             boxes[0][2] + _LABEL_MARGIN_MM, boxes[0][3] + _LABEL_MARGIN_MM)
    assert _box_overlap(grown, boxes[1]) == 0.0


def test_a_label_stays_nearer_its_own_vertex_than_any_other():
    """A label closer to a neighbour than to its owner names the wrong vertex.

    Clearance alone does not catch this: the label can be clear of every line
    and every dot and still sit between two vertices, reading as either.  The
    placement therefore also charges for a foreign vertex that comes within
    1.15x of the owner's own distance.  Inert on the current figures -- their
    vertices are never that close -- and a guard against the case that is.
    """
    import math
    lone = place_labels({'a': (0.0, 0.0, 3.5, 2.2)}, [], [(0.0, 0.0)])
    crowd = place_labels({'a': (0.0, 0.0, 3.5, 2.2)}, [],
                         [(0.0, 0.0), (0.0, 4.6)])
    assert lone['a'] == (90, 1.0), 'the canonical place, with room'
    assert crowd['a'] != lone['a'], 'a neighbour overhead must push it off'
    box = _label_box(0.0, 0.0, *crowd['a'], 3.5, 2.2, 0.825)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    assert math.hypot(cx, cy) * 1.15 <= math.hypot(cx, cy - 4.6)

    # "Distant" has to be measured against where the glyphs PRINT.  A 2.2 mm
    # label at 1pt sits 3.63 mm out once the node's own padding is counted,
    # so a dot 6 mm up is only 2.4 mm away -- nearer than the owner, and the
    # placement is right to move.  9 mm is distant; 6 mm never was.
    far = place_labels({'a': (0.0, 0.0, 3.5, 2.2)}, [],
                       [(0.0, 0.0), (0.0, 9.0)])
    assert far['a'] == (90, 1.0), 'a distant neighbour must not disturb it'


def test_a_label_box_is_where_tikz_actually_prints_the_glyphs():
    """``label distance`` is measured from the NODE border, not from the ink.

    TikZ pads a label node by ``inner sep`` (0.3333em) plus ``outer sep``
    before deciding where its border is, so the text lands that much further
    out than a naive text-box model predicts.  Checked against
    ``\\pgfpointanchor`` on the 66 printed two-loop panels, the omission cost
    +1.0 mm at a cardinal angle and +2.0 mm at a diagonal -- enough to print
    seven labels across a propagator the placement had scored as clear.
    """
    import math
    w, h, dot_r, dist = 3.5, 2.2, 0.825, 1.0
    box = _label_box(0.0, 0.0, 90, dist, w, h, dot_r)
    cy = (box[1] + box[3]) / 2.0
    assert math.isclose(cy, dot_r + dist * _PT_MM + h / 2.0 + _INNER_SEP_MM,
                        rel_tol=1e-9), 'the node padding is part of the reach'
    # The box handed back is still the size of the INK: padding moves the
    # label, it is not something the drawing has to be kept clear of.
    assert math.isclose(box[2] - box[0], w, rel_tol=1e-9)
    assert math.isclose(box[3] - box[1], h, rel_tol=1e-9)
    # A diagonal reaches further than a cardinal, and by more than the text
    # box alone would say -- which is why the two score honestly together.
    diag = _label_box(0.0, 0.0, 45, dist, w, h, dot_r)
    dcx, dcy = (diag[0] + diag[2]) / 2.0, (diag[1] + diag[3]) / 2.0
    assert math.hypot(dcx, dcy) > cy


def _labelled(pd):
    """A prediagram dressed with real vertex types, so labels are emitted."""
    from engine.core.vertices import SourceType, VertexType
    D = pd[0]
    leaf = set(pd[2])
    assign = {}
    for v in D.vertices():
        if v in leaf:
            continue
        if D.in_degree(v) == 0:
            assign[v] = SourceType('-D', [('xt', 1)] * D.out_degree(v), (2, 0))
        else:
            assign[v] = VertexType('eps', [('xt', 1)],
                                   [('dx', 1)] * max(1, D.out_degree(v)),
                                   (1, 3))

    class _TD:
        prediagram = pd
        vertex_assignments = assign
        external_legs = {v: ('dx', 1) for v in leaf}
    return _TD()


def _label_boxes(tex, mm):
    """Every label in an emitted picture, as a printed-mm box."""
    import re
    out = []
    for ln in tex.splitlines():
        m = re.search(r'at \(([-\d.]+), ([-\d.]+)\)', ln)
        lm = re.search(r'label distance=([\d.]+)pt\]\s*(-?\d+):'
                       r'\\\((.*?)\\\)\}', ln)
        if not (m and lm):
            continue
        w, h = _label_extent_mm(lm.group(3))
        out.append(_label_box(float(m.group(1)) * mm, float(m.group(2)) * mm,
                              int(lm.group(2)), float(lm.group(1)),
                              w, h, DOT_RADIUS_MM))
    return out


def test_no_propagator_is_drawn_through_a_label(prediagrams_2_2):
    """The defect that survived the first placement pass.

    Labels were scored by the clearance of their ANCHOR POINT, but a label is
    a BOX -- and one that does not shrink with the panel, since `scale=`
    moves coordinates and not node text.  A `v_2` prints 3.5 mm wide inside a
    46 mm panel, so a point could be 'clear' while two thirds of the text lay
    across a propagator.  Measured over the 66 printed two-loop panels, 33 mm
    of propagator ran through label text on 6 of them.
    """
    for pd in prediagrams_2_2[:60]:
        dia = _labelled(pd)
        pos = layout_typed_diagram(pd)
        mm = mm_per_unit(pos)
        tex = to_tikz_feynman(dia, symbolic_factors=True,
                              propagator_label=None)
        boxes = _label_boxes(tex, mm)
        edges = sorted(pd[0].edges(labels=False))
        bends = edge_bends(edges, 14)
        for (u, v), b in zip(edges, bends):
            poly = [tuple(c * mm for c in path_point(pos[u], pos[v], b,
                                                     k / 12.0))
                    for k in range(13)]
            for box in boxes:
                inside = sum(_seg_in_box(p, q, box)
                             for p, q in zip(poly, poly[1:]))
                assert inside < 0.5, (
                    f'{inside:.2f} mm of propagator {u}->{v} runs through a '
                    f'label')


def test_labels_do_not_collide_with_each_other_or_with_a_vertex(
        prediagrams_2_2):
    """A label touching the wrong dot names the wrong vertex."""
    for pd in prediagrams_2_2[:60]:
        dia = _labelled(pd)
        pos = layout_typed_diagram(pd)
        mm = mm_per_unit(pos)
        boxes = _label_boxes(to_tikz_feynman(dia, symbolic_factors=True,
                                             propagator_label=None), mm)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert _box_overlap(boxes[i], boxes[j]) == 0.0
            for x, y in pos.values():
                r = DOT_RADIUS_MM
                assert _box_overlap(boxes[i], (x * mm - r, y * mm - r,
                                               x * mm + r, y * mm + r)) == 0.0


def test_no_label_is_printed_through_an_arrowhead(prediagrams_2_2):
    """A propagator is a line to the collision test, but its arrow is a blob.

    Scoring a label against the zero-width polyline alone lets it be placed
    just clear of the line and straight through the 1.91 mm arrowhead -- seen
    on panel 17 of the two-loop figure, where the head cut into the `v` of a
    `v_2`.  The heads are obstacles in their own right.
    """
    half = ARROWHEAD_MM / 2.0
    for pd in prediagrams_2_2[:60]:
        dia = _labelled(pd)
        pos = layout_typed_diagram(pd)
        mm = mm_per_unit(pos)
        boxes = _label_boxes(to_tikz_feynman(dia, symbolic_factors=True,
                                             propagator_label=None), mm)
        edges = sorted(pd[0].edges(labels=False))
        bends = edge_bends(edges, 14, pos, mm)
        for (u, v), b, t in zip(edges, bends,
                                arrow_positions(pos, edges, bends, mm)):
            q = path_point(pos[u], pos[v], b, t)
            head = (q[0] * mm - half, q[1] * mm - half,
                    q[0] * mm + half, q[1] * mm + half)
            for box in boxes:
                assert _box_overlap(box, head) == 0.0, (
                    f'a label is printed over the arrowhead of {u}->{v}')


# ── one stroke, one meaning ─────────────────────────────────────────

class _Styled:
    """A diagram carrying only the edge typing the style map reads."""

    def __init__(self, *pairs):
        self.edge_types = {i: p for i, p in enumerate(pairs)}


_XX = (('xt', 1), ('dx', 1))
_XY = (('xt', 1), ('dy', 1))
_YY = (('yt', 1), ('dy', 1))


def test_a_stroke_means_one_component_across_the_whole_figure():
    """Line styles are a symbol map, and break the same way per-panel.

    Handed out over the pairings each panel happens to contain, the FIRST
    style goes to whatever that panel has first -- so a solid line means
    G_xx in one panel and G_xy in the next, while the key claims a single
    meaning for the figure.  Measured on the 24-panel two-field figure,
    `solid` covered three different components and `dashed` three more.
    """
    from engine.diagrams.tikz_export import edge_style_map
    a, b = _Styled(_XX, _XY), _Styled(_XX, _YY)
    per_panel = (edge_style_map(a), edge_style_map(b))
    assert per_panel[0][('x', 'y')] == per_panel[1][('y', 'y')], (
        'the per-panel collision this test exists to prevent is gone; '
        'rewrite the test rather than deleting it')

    shared = edge_style_map([a, b])
    assert len({shared[('x', 'x')], shared[('x', 'y')],
                shared[('y', 'y')]}) == 3
    for dia in (a, b):
        for pair in {(_field_pair_of(k)) for k in dia.edge_types.values()}:
            assert shared[pair] == shared[pair]


def _field_pair_of(legs):
    from engine.diagrams.tikz_export import _field_symbol
    return (_field_symbol(legs[0]), _field_symbol(legs[1]))


def test_a_shared_style_map_overrides_the_per_panel_one(prediagrams_2_1):
    """A panel must draw with the FIGURE's map, not with its own.

    On its own this diagram carries one component, so its per-panel map
    would give it the first style -- solid -- exactly as some other panel's
    G_xx.  Handed the figure-wide map it must draw the stroke that means
    G_yy everywhere in the figure.
    """
    from engine.diagrams.tikz_export import edge_style_map
    pd = prediagrams_2_1[0]

    from collections import Counter
    _copies = Counter()

    def _keys():
        for u, v in pd[0].edges(labels=False):
            yield (u, v, _copies[(u, v)])
            _copies[(u, v)] += 1

    class _TD:
        prediagram = pd
        edge_types = {k: _YY for k in _keys()}

    shared = edge_style_map([_Styled(_XX, _XY), _Styled(_XX, _YY)])
    want = shared[('y', 'y')]
    assert want, 'G_yy should not be the unstyled default here'
    alone = to_tikz_feynman(_TD(), style_by_field=True)
    figure = to_tikz_feynman(_TD(), style_by_field=True, edge_styles=shared)
    drawn = [ln for ln in figure.splitlines() if ' -- [' in ln]
    assert drawn and all(want in ln for ln in drawn)
    assert not any(want in ln for ln in alone.splitlines() if ' -- [' in ln)


# ── source vs interaction naming ────────────────────────────────────

def _real_types():
    from engine.core.vertices import SourceType, VertexType
    return SourceType, VertexType


def _fig(*assignments):
    """A minimal diagram carrying just the vertex types, for the symbol map."""
    class _TD:
        prediagram = None
        vertex_assignments = dict(enumerate(assignments))
    return _TD()


def test_source_is_named_by_its_leg_count():
    """A noise source IS its cumulant order; an opaque index would hide it."""
    from engine.diagrams.tikz_export import vertex_symbol_map
    SourceType, _V = _real_types()
    two = SourceType('-D', [('xt', 1)] * 2, (2, 0))
    three = SourceType('K', [('xt', 1)] * 3, (3, 0))
    m = vertex_symbol_map(_fig(two, three))
    assert m['-D'] == r'\kappa_{2}'
    assert m['K'] == r'\kappa_{3}'


def test_sources_do_not_consume_an_interaction_index():
    """v_j runs over the DISTINCT INTERACTION factors only."""
    from engine.diagrams.tikz_export import vertex_symbol_map
    SourceType, VertexType = _real_types()
    src = SourceType('-D', [('xt', 1)] * 2, (2, 0))
    a = VertexType('g', [('xt', 1)], [('dx', 1)] * 3, (1, 3))
    b = VertexType('h', [('xt', 1)], [('dx', 1)] * 2, (1, 2))
    m = vertex_symbol_map(_fig(src, a, b))
    assert m['g'] == r'v_{1}' and m['h'] == r'v_{2}'
    assert m['-D'] == r'\kappa_{2}'


def test_a_bare_factor_object_is_still_an_interaction():
    """Nothing declaring itself a source must not be named as one."""
    from engine.diagrams.tikz_export import vertex_symbol_map

    class _C:
        def _latex_(self): return 'A'

    class _V:
        coefficient = _C()

    assert vertex_symbol_map(_fig(_V()))['A'] == r'v_{1}'


def test_source_and_interaction_stay_apart_when_expressions_agree():
    """Same LaTeX, different KIND -> different symbol; a tex-keyed map alone
    could not tell them apart."""
    from engine.diagrams.tikz_export import vertex_symbol_map, _symbol_for
    SourceType, VertexType = _real_types()
    src = SourceType('D', [('xt', 1)] * 2, (2, 0))
    vtx = VertexType('D', [('xt', 1)], [('dx', 1)] * 2, (1, 2))
    m = vertex_symbol_map(_fig(src, vtx), keyed=True)
    assert set(m.values()) == {r'\kappa_{2}', r'v_{1}'}
    assert _symbol_for(m, src, 'D') == r'\kappa_{2}'
    assert _symbol_for(m, vtx, 'D') == r'v_{1}'


def test_same_order_sources_of_different_fields_are_distinguishable():
    """Two 2-leg noises would both want \\kappa_2; the key must stay usable."""
    from engine.diagrams.tikz_export import vertex_symbol_map
    SourceType, _V = _real_types()
    dx = SourceType('-D_1', [('xt', 1)] * 2, (2, 0))
    dy = SourceType('-D_2', [('yt', 1)] * 2, (2, 0))
    m = vertex_symbol_map(_fig(dx, dy))
    assert m['-D_1'] != m['-D_2']
    assert m['-D_1'].startswith(r'\kappa_{2}')
    assert 'x' in m['-D_1'] and 'y' in m['-D_2']


def test_a_lone_source_keeps_the_plain_symbol():
    """The disambiguating superscript must fire only on a real collision."""
    from engine.diagrams.tikz_export import vertex_symbol_map
    SourceType, _V = _real_types()
    m = vertex_symbol_map(_fig(SourceType('-D', [('xt', 1)] * 2, (2, 0))))
    assert m['-D'] == r'\kappa_{2}'


def test_drawn_panel_labels_sources_with_kappa(prediagrams_2_1):
    """End to end: the picture itself must carry the source symbols."""
    SourceType, VertexType = _real_types()
    D, G, leaves, internal = prediagrams_2_1[0]
    src = SourceType('-D', [('xt', 1)] * 2, (2, 0))
    vtx = VertexType('g', [('xt', 1)], [('dx', 1)] * 3, (1, 3))

    class _TD:
        prediagram = (D, G, leaves, internal)
        vertex_assignments = {v: (src if i == 0 else vtx)
                              for i, v in enumerate(sorted(internal))}

    tex = to_tikz_feynman(_TD(), symbolic_factors=True)
    assert r'\kappa_{2}' in tex
    assert r'v_{1}' in tex
    assert '-D' not in tex, 'the expression belongs in the key, not the panel'


# ── the drawing must not invent structure it does not have ──────────

def test_no_arrowhead_is_drawn_on_a_propagator_it_does_not_belong_to():
    """A solid triangle on a line crossing reads as a VERTEX.

    The layered layout is symmetric, so two edges between the same pair of
    layers cross at BOTH their midpoints; separating the two ARROWHEADS from
    each other still leaves one of them centred on the crossing, sitting on
    the other propagator.  Measured over the 66 printed two-loop panels: 18
    such overlaps on 18 panels, the worst with the head's centre 0.00 mm from
    the foreign line.  Foreign propagators are therefore obstacles too.
    """
    import math
    from engine.diagrams.tikz_export import (arrow_positions, path_point,
                                             edge_bends)
    from engine.diagrams.typed_diagram_layout import (ARROWHEAD_MM,
                                                      mm_per_unit)
    pos = {'a': (0.0, 1.0), 'b': (2.3, -1.0), 'c': (0.0, -1.0), 'd': (2.3, 1.0)}
    edges = [('a', 'b'), ('c', 'd')]
    bends = edge_bends(edges, 14)
    mm = mm_per_unit(pos)
    ts = arrow_positions(pos, edges, bends, mm)
    assert any(abs(t - 0.5) > 1e-9 for t in ts), 'the midpoint IS the crossing'

    def clearance(i, j, t):
        q = path_point(pos[edges[i][0]], pos[edges[i][1]], bends[i], t)
        P = [path_point(pos[edges[j][0]], pos[edges[j][1]], bends[j], k / 40.0)
             for k in range(41)]
        return min(math.hypot(q[0] - p[0], q[1] - p[1]) for p in P) * mm

    for i, j in ((0, 1), (1, 0)):
        assert clearance(i, j, 0.5) < ARROWHEAD_MM / 2.0, 'premise: midpoints clash'
        assert clearance(i, j, ts[i]) >= ARROWHEAD_MM / 2.0, (
            'arrowhead %d still drawn on the other propagator' % i)


def test_two_arrowheads_are_separated_not_merely_tangent():
    """Touching triangles print as one blob, which is what a bubble did.

    A `fermion` head measures 1.905 mm both along and across the line, so a
    rule of `centre distance >= ARROWHEAD_MM` is met exactly when two heads
    touch corner to corner.  The two heads of a bubble sat at the same point
    along their arcs, one lens-width apart, and the lens is opened to
    MIN_BUBBLE_MM = 2.0 mm -- so they grazed by 0.04 mm and filled the bubble
    with a single black bowtie.
    """
    import math
    from engine.diagrams.tikz_export import (arrow_positions, path_point,
                                             edge_bends)
    from engine.diagrams.typed_diagram_layout import (ARROWHEAD_MM,
                                                      mm_per_unit)
    pos = {'u': (0.0, 0.0), 'v': (2.3, 0.0)}
    edges = [('u', 'v'), ('u', 'v')]
    mm = mm_per_unit(pos)
    bends = edge_bends(edges, 14, pos, mm)
    ts = arrow_positions(pos, edges, bends, mm)
    heads = [path_point(pos['u'], pos['v'], b, t) for b, t in zip(bends, ts)]
    gap = math.hypot(heads[0][0] - heads[1][0],
                     heads[0][1] - heads[1][1]) * mm
    assert gap > ARROWHEAD_MM + 0.3, (
        'the two heads of a bubble are %.2f mm apart, %.2f mm of ink'
        % (gap, ARROWHEAD_MM))


def test_a_bubble_is_judged_by_its_ARC_and_not_by_its_chord():
    """Every clearance used to be measured on the straight chord.

    A parallel pair is drawn bowed, so its ink sweeps a lens around the chord
    -- and a vertex clear of the chord can be sitting on the arc.  Found on
    three-loop panel 479 of the quartic OU set: the guard scored the drawing
    a PERFECT zero while an arc passed 0.04 mm from an unrelated vertex.
    """
    from engine.diagrams.typed_diagram_layout import (
        vertex_edge_clearances, legibility_penalty, mm_per_unit,
        bubble_half_width_mm, DOT_RADIUS_MM, _seg_distance)
    pos = {'u': (0.0, 0.0), 'v': (4.0, 0.0), 'w': (2.0, 0.5)}
    pair = [('u', 'v'), ('u', 'v')]
    single = [('u', 'v')]
    mm = mm_per_unit(pos)
    chord = _seg_distance(pos['w'], pos['u'], pos['v']) * mm
    assert chord > 2 * DOT_RADIUS_MM, 'premise: clear of the CHORD'
    assert bubble_half_width_mm(4.0 * mm) > 0.0, 'premise: the pair is bowed'

    def worst(edges):
        return min(d for d, _v, _e in vertex_edge_clearances(pos, edges, mm))

    assert worst(single) == chord, 'a lone edge is still its chord'
    assert worst(pair) < 2 * DOT_RADIUS_MM, 'the ARC runs through the dot'
    assert legibility_penalty(pos, single) == 0.0
    assert legibility_penalty(pos, pair) > 0.0, (
        'a bubble sweeping a vertex must cost the layout something')

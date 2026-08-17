"""Layout geometry and tikz-feynman emission.

The emitter has no numerical content, so these are structural checks: that
time runs right to left, that parallel edges are bowed apart rather than
drawn on top of each other, and that the paper's vertex conventions survive
into the output.
"""

import pytest

import engine.enumeration.loop_diagram_enumeration as L
from engine.diagrams.typed_diagram_layout import (
    causal_depths, layout_typed_diagram,
)
from engine.diagrams.tikz_export import (
    to_tikz_feynman, diagram_to_standalone, DEFAULT_SYMBOL_MAP,
)


@pytest.fixture(scope='module')
def prediagrams_2_1():
    return L.enumerate_all(2, 1, verbose=False)[2]


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
    assert tex.count('[empty dot]') == len(leaves), 'externals unshaded'
    assert tex.count('[dot]') == len(internal), 'sources/interactions solid'
    assert r'\delta\phi(y_{1})' in tex


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
        assert 'half left' in tex and 'half right' in tex, (
            'parallel propagators must be bent in opposite directions')
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
    assert r'\varepsilon' in out and r'\phi^{*}' in out
    assert 'mathit' not in out


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

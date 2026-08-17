"""Panel-array assembly.

Structural checks only -- the per-diagram drawing is covered by
``test_tikz_export``.  The one behavioural rule worth pinning is that
truncation is never silent: a figure showing a subset of the diagram set
must say so, or it misrepresents the expansion.
"""

import pytest

import engine.enumeration.loop_diagram_enumeration as L
from engine.diagrams.tikz_figure import (
    diagrams_to_figure, diagrams_to_standalone, legend_table,
)


@pytest.fixture(scope='module')
def pds():
    return L.enumerate_all(2, 1, verbose=False)[2]


def test_one_panel_per_diagram(pds):
    tex = diagrams_to_figure(pds[:4], ncol=2)
    assert tex.count(r'\subcaptionbox') == 4
    assert tex.count(r'\begin{tikzpicture}') == 4


def test_row_breaks_follow_ncol(pds):
    # 4 panels at ncol=2 -> exactly one break (after the 2nd); none trailing
    assert diagrams_to_figure(pds[:4], ncol=2).count(r'\\[1.2em]') == 1
    assert diagrams_to_figure(pds[:4], ncol=4).count(r'\\[1.2em]') == 0


def test_truncation_is_announced(pds):
    """A subset must never be presented as the whole set."""
    tex = diagrams_to_figure(pds, ncol=3, max_panels=2)
    assert tex.count(r'\subcaptionbox') == 2
    assert 'omitted' in tex
    assert str(len(pds)) in tex, 'the true total must appear'


def test_no_truncation_note_when_all_shown(pds):
    assert 'omitted' not in diagrams_to_figure(pds, max_panels=len(pds))
    assert 'omitted' not in diagrams_to_figure(pds)


def test_caption_and_label_are_optional(pds):
    bare = diagrams_to_figure(pds[:2])
    assert r'\caption' not in bare and r'\label' not in bare
    full = diagrams_to_figure(pds[:2], caption='Two diagrams.', label='fig:x')
    assert r'\caption{Two diagrams.}' in full and r'\label{fig:x}' in full


def test_array_suppresses_edge_labels_by_default(pds):
    """An array is structural; a lone diagram is pedagogical."""
    from engine.diagrams.tikz_export import to_tikz_feynman
    assert 'edge label' not in diagrams_to_figure(pds[:2])
    assert 'edge label' in to_tikz_feynman(pds[0])
    assert 'edge label' in diagrams_to_figure(pds[:2], propagator_label='G')


def test_panel_caption_carries_loop_order_and_multiplicity(pds):
    recs = [{'typed_diagram': pds[0], 'ell': 0, 'multiplicity': 1},
            {'typed_diagram': pds[1], 'ell': 1, 'multiplicity': 3}]
    tex = diagrams_to_figure(recs)
    assert r'$\ell=0$' in tex and r'$\ell=1$' in tex
    assert '$M=1$' in tex and '$M=3$' in tex


def test_standalone_supplies_a_float_context(pds):
    """subcaptionbox refuses to run outside a float."""
    doc = diagrams_to_standalone(pds[:2], ncol=2)
    assert r'\captionsetup{type=figure}' in doc
    assert r'\begin{figure}' not in doc, 'standalone has no figure environment'
    assert doc.rstrip().endswith(r'\end{document}')


@pytest.mark.slow
def test_panel_array_compiles(tmp_path, pds):
    import shutil, subprocess
    if shutil.which('pdflatex') is None:
        pytest.skip('pdflatex not installed')
    src = tmp_path / 'f.tex'
    src.write_text(diagrams_to_standalone(pds[:4], ncol=2))
    proc = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'f.tex'],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 and 'tikz-feynman' in (proc.stdout + proc.stderr):
        pytest.skip('tikz-feynman package not installed')
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert (tmp_path / 'f.pdf').exists()


# ── public entry point ──────────────────────────────────────────────

def test_export_tikz_accepts_bare_records(pds):
    """dd.export_tikz should take records, a result dict, or one diagram."""
    import daedalus as dd
    recs = [{'typed_diagram': p, 'ell': 1, 'multiplicity': 1} for p in pds[:3]]

    arr = dd.export_tikz(recs)
    assert arr.count(r'\subcaptionbox') == 3

    one = dd.export_tikz(recs, index=1)
    assert one.count(r'\begin{tikzpicture}') == 1
    assert r'\subcaptionbox' not in one

    as_result = dd.export_tikz({'diagrams': recs})
    assert as_result.count(r'\subcaptionbox') == 3


def test_export_tikz_single_labels_array_does_not(pds):
    import daedalus as dd
    recs = [{'typed_diagram': p, 'ell': 1} for p in pds[:2]]
    assert 'edge label' in dd.export_tikz(recs, index=0)
    assert 'edge label' not in dd.export_tikz(recs)


def test_export_tikz_ell_filter(pds):
    import daedalus as dd
    recs = ([{'typed_diagram': pds[0], 'ell': 0}]
            + [{'typed_diagram': p, 'ell': 1} for p in pds[1:4]])
    assert dd.export_tikz(recs, ell=1).count(r'\subcaptionbox') == 3
    assert dd.export_tikz(recs, ell=0).count(r'\subcaptionbox') == 1
    with pytest.raises(ValueError, match='no diagrams at ell=7'):
        dd.export_tikz(recs, ell=7)


def test_export_tikz_writes_standalone_for_tex_path(tmp_path, pds):
    """A .tex path must be compilable on its own, without the caller
    remembering to ask for standalone."""
    import daedalus as dd
    recs = [{'typed_diagram': p, 'ell': 1} for p in pds[:2]]
    out = tmp_path / 'd.tex'
    dd.export_tikz(recs, path=str(out))
    txt = out.read_text()
    assert txt.startswith(r'\documentclass')
    assert txt.rstrip().endswith(r'\end{document}')


# ── key table and pagination ────────────────────────────────────────

def test_symbol_map_is_shared_across_the_figure():
    """v_1 must mean the same factor in every panel, and the key must list
    every factor the figure uses -- not just the first panel's."""
    from engine.diagrams.tikz_export import vertex_symbol_map

    class _C:
        def __init__(self, t): self.t = t
        def _latex_(self): return self.t

    class _V:
        def __init__(self, t): self.coefficient = _C(t)

    class _TD:
        def __init__(self, ts):
            self.prediagram = None
            self.vertex_assignments = {i: _V(t) for i, t in enumerate(ts)}

    a, b = _TD(['A', 'B']), _TD(['B', 'C'])
    per_panel_a = vertex_symbol_map(a)
    shared = vertex_symbol_map([a, b])
    assert set(shared) == {'A', 'B', 'C'}, 'key must cover every panel'
    # 'B' keeps its symbol when the map is built over both panels
    assert shared['B'] == per_panel_a['B']
    assert len(set(shared.values())) == 3, 'symbols must be distinct'


def test_legend_lists_styles_and_symbols(pds):
    from engine.diagrams.tikz_figure import legend_table
    tex = legend_table(pds[0], vertex_symbols={r'-D': r'v_{1}'})
    assert 'tabular' in tex
    assert r'v_{1}' in tex and '-D' in tex


def test_pages_output_is_not_a_float(pds):
    """A complete set must flow across pages; a figure float cannot break."""
    from engine.diagrams.tikz_figure import diagrams_to_pages
    tex = diagrams_to_pages(pds, ncol=2, caption='All of them.')
    assert r'\begin{figure}' not in tex, 'must not be a float'
    assert r'\captionof{figure}' in tex, 'caption without a float'
    assert tex.count(r'\begin{minipage}') == len(pds), 'every diagram drawn'


def test_pages_draws_everything_with_no_truncation(pds):
    from engine.diagrams.tikz_figure import diagrams_to_pages
    tex = diagrams_to_pages(pds)
    assert 'omitted' not in tex
    assert tex.count(r'\begin{tikzpicture}') == len(pds)


def test_legend_lists_sources_and_interactions_with_expressions(pds):
    """Both kinds of vertex appear on the drawings, so both must be in the
    key -- a symbol the key omits is unreadable."""
    from engine.diagrams.tikz_figure import legend_table
    from engine.core.vertices import SourceType, VertexType

    class _TD:
        prediagram = None
        vertex_assignments = {
            0: SourceType('-D', [('xt', 1)] * 2, (2, 0)),
            1: VertexType('g', [('xt', 1)], [('dx', 1)] * 3, (1, 3)),
        }

    tex = legend_table(_TD(), edges=False)
    assert r'\(\kappa_{2}\) & \(-D\)' in tex
    assert r'\(v_{1}\) & \(g\)' in tex
    # interactions first, then the noise cumulants: the key reads as blocks
    assert tex.index(r'v_{1}') < tex.index(r'\kappa_{2}')


def test_pages_share_one_map_across_sources_and_interactions():
    """kappa_2 and v_1 must mean the same factor in every panel."""
    from engine.diagrams.tikz_figure import diagrams_to_pages
    from engine.core.vertices import SourceType, VertexType
    import engine.enumeration.loop_diagram_enumeration as L

    pd = L.enumerate_all(2, 1, verbose=False)[2][0]
    D, G, leaves, internal = pd
    src = SourceType('-D', [('xt', 1)] * 2, (2, 0))
    a = VertexType('g', [('xt', 1)], [('dx', 1)] * 3, (1, 3))
    b = VertexType('h', [('xt', 1)], [('dx', 1)] * 2, (1, 2))

    def td(types):
        class _TD:
            prediagram = pd
            vertex_assignments = {v: t for v, t in
                                  zip(sorted(internal), types)}
        return _TD()

    # panel 1 never shows 'h'; panel 2 leads with it.  Numbered per panel,
    # 'h' would be v_1 in panel 2 and v_2 overall.
    n = len(internal)
    p1 = td([src] + [a] * (n - 1))
    p2 = td([b] * n)
    tex = diagrams_to_pages([p1, p2], ncol=2, symbolic_factors=True,
                            legend_from=p1)
    assert r'\kappa_{2}' in tex
    assert r'v_{1}' in tex and r'v_{2}' in tex
    # the key covers BOTH panels, not just the one it was built from
    assert r'\(v_{2}\) & \(h\)' in tex


def test_the_key_describes_the_whole_figure_not_one_panel():
    """`legend_from` names ONE diagram; the key must still cover them all.

    Derived from that diagram alone the key lists only the components that
    panel contains, so a stroke used elsewhere in the figure appears in the
    drawing with nothing in the key to explain it.  Passing the figure-wide
    style map fixes both halves at once.
    """
    from engine.diagrams.tikz_export import edge_style_map

    class _Styled:
        def __init__(self, *pairs):
            self.edge_types = {i: p for i, p in enumerate(pairs)}

    xx = (('xt', 1), ('dx', 1))
    xy = (('xt', 1), ('dy', 1))
    yy = (('yt', 1), ('dy', 1))
    one_panel, whole = _Styled(xx), [_Styled(xx), _Styled(xy), _Styled(yy)]

    narrow = legend_table(one_panel, vertices=False)
    assert narrow.count(r'\\') == 1, 'one panel knows only its own component'

    wide = legend_table(one_panel, vertices=False,
                        edge_styles=edge_style_map(whole))
    assert wide.count(r'\\') == 3
    for name in ('G_{x}', 'G_{xy}', 'G_{y}'):
        assert name in wide

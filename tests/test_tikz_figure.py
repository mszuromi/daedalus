"""Panel-array assembly.

Structural checks only -- the per-diagram drawing is covered by
``test_tikz_export``.  The one behavioural rule worth pinning is that
truncation is never silent: a figure showing a subset of the diagram set
must say so, or it misrepresents the expansion.
"""

import pytest

import engine.enumeration.loop_diagram_enumeration as L
from engine.diagrams.tikz_figure import (
    diagrams_to_figure, diagrams_to_standalone,
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

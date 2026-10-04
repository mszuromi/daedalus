"""
api.report — multi-page PDF showing prediagrams, typed-diagram
assignments, and per-diagram numerical contributions.

The report is meant for *intuition-building*:  scroll through diagrams,
see what the cumulant slice looks like, see how each diagram contributes
to the total.

generate_report() produces:
  - a cover page (model name, parameters, MF values, total slice plot)
  - one page per typed diagram with:
      - prediagram graph rendered via networkx + matplotlib
      - vertex assignments table
      - edge propagator labels
      - per-diagram contribution C_Γ(τ) line plot

The diagram pages come in a canonical order (loop order, then the
diagram's isomorphism class) and show canonical vertex numbers, so a page
depends only on the diagram's class: not on the order of
``result['diagrams']`` nor on which labelled member of the class the
prediagram source (cache file, in-memory streaming, eager enumerator)
happened to provide.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from api.compute import compute_cumulants


# ───────────────────────────────────────────────────────────────────────
# matplotlib PDF backend monkey-patch
# ───────────────────────────────────────────────────────────────────────
# Defensive cleanups in this file (Sage Integer → Python int casts in
# _draw_prediagram, plt.close('all') around PdfPages, ...) catch most
# Sage objects before they reach matplotlib.  But matplotlib's text
# layout / mathtext path can still pull in Sage RealLiteral values via
# layout calls (tight_layout uses the renderer's text-metric machinery)
# in a Jupyter-notebook context that has previously rendered Sage
# expressions inline — those leak into the per-Figure alphaStates dict
# and only surface at PdfPages.finalize() → writeExtGSTates() →
# pdfRepr(dict) with
#
#     TypeError: Don't know a PDF representation for
#                <class 'sage.rings.real_mpfr.RealLiteral'> objects
#
# We can't reach into matplotlib's text-metric cache, but we can teach
# its pdfRepr() to coerce Sage RealLiteral (and any Sage type with a
# __float__) to a plain Python float before serialization.  This is
# applied once at module import; subsequent generate_report() calls
# pick it up automatically.
def _install_pdf_repr_sage_fallback():
    from matplotlib.backends import backend_pdf as _bp
    _orig_pdfRepr = _bp.pdfRepr

    def _patched(obj):
        try:
            return _orig_pdfRepr(obj)
        except TypeError:
            # Last resort: any object that quacks like a real number.
            # Sage RealLiteral has __float__; SR scalars do too.
            try:
                return _orig_pdfRepr(float(obj))
            except (TypeError, ValueError):
                pass
            try:
                return _orig_pdfRepr(complex(obj))
            except (TypeError, ValueError):
                pass
            raise

    if getattr(_bp.pdfRepr, '__name__', '') != '_patched':
        _bp.pdfRepr = _patched

_install_pdf_repr_sage_fallback()


def _canonical_view(td):
    """``(class_key, display_id)`` of a typed diagram, both functions of its
    isomorphism class alone.

    ``class_key`` is ``str(diagram_signature(td))`` (a complete invariant:
    the canonical form of the coloured incidence digraph, leaves coloured
    by field).  ``display_id`` maps every vertex to the number the report
    prints: leaves first, then internal vertices, each in the order of the
    same canonical labelling.  Two labelled representatives of one class
    therefore get the same page.
    """
    from engine.diagrams.symmetry import _colored_incidence_digraph
    D, _, _, color_groups = _colored_incidence_digraph(td, fix_external=False)
    # Same partition and canonical labelling as ``diagram_signature``.
    keys = sorted(color_groups.keys(), key=str)
    partition = [color_groups[key] for key in keys]
    C, cert = D.canonical_label(partition=partition, certificate=True)
    cells = tuple(tuple(sorted(cert[v] for v in color_groups[key]))
                  for key in keys)
    signature = (tuple(str(key) for key in keys), cells,
                 tuple(sorted(C.edges(labels=False))))
    leaves = set(td.external_legs)
    order = sorted(td.prediagram[0].vertices(),
                   key=lambda v: (v not in leaves, cert[('V', v)]))
    return str(signature), {v: i for i, v in enumerate(order)}


def _prediagram_layout(td, display_id=None):
    """What :func:`_draw_prediagram` draws, as plain data.

    Returns ``{'nodes': [(id, (x, y), colour), ...], 'edges': [(u, v), ...],
    'edge_labels': {(u, v): text}}`` with Python ints and floats only (Sage
    Integers leak into matplotlib's PDF state otherwise; see
    :func:`_draw_prediagram`).  Leaves sit at the top (y=1), internal
    vertices at the bottom (y=0); colours are green (leaf), red
    (interaction), blue (source).  An edge label is the edge's
    (resp_leg, phys_leg) propagator pair, or the distinct pairs, sorted,
    when several edges join the same two vertices.

    ``display_id`` (vertex -> printed number, from :func:`_canonical_view`)
    fixes the vertex numbers and every order; without it the diagram's own
    labels are used.
    """
    D = td.prediagram[0]
    if display_id is None:
        _vmap = {v: int(v) for v in D.vertices()}
    else:
        _vmap = {v: int(display_id[v]) for v in D.vertices()}
    leaves = [_vmap[v] for v in td.prediagram[2]]
    if display_id is not None:
        leaves = sorted(leaves)
    leaf_set = set(leaves)
    verts = sorted(D.vertices(), key=lambda v: _vmap[v])

    def _edge_text(u, v, lbl):
        et = td.edge_types.get((u, v, lbl))
        if et is None:
            return None
        resp_leg, phys_leg = et
        return (rf'${resp_leg[0]}_{int(resp_leg[1])}\!\to\!'
                rf'\,{phys_leg[0]}_{int(phys_leg[1])}$')
    edges = sorted(((_vmap[u], _vmap[v], _edge_text(u, v, lbl))
                    for u, v, lbl in D.edges()),
                   key=lambda e: (e[0], e[1], e[2] or ''))

    pos = {}
    leaf_xs = np.linspace(0.0, 1.0, max(len(leaves), 2))
    for j, lf in enumerate(leaves):
        pos[lf] = (float(leaf_xs[j]), 1.0)
    internal = [_vmap[v] for v in verts if _vmap[v] not in leaf_set]
    int_xs = np.linspace(0.2, 0.8, max(len(internal), 1))
    for j, v in enumerate(internal):
        pos[v] = (float(int_xs[j]), 0.0)

    nodes = []
    for v_sage in verts:
        v = _vmap[v_sage]
        if v in leaf_set:
            colour = '#2ECC71'                   # green = leaf
        else:
            vt = td.vertex_assignments.get(v_sage)
            if vt is None:
                colour = '#999999'
            elif hasattr(vt, 'physical_legs'):
                colour = '#E74C3C'               # red = interaction
            else:
                colour = '#3498DB'               # blue = source
        nodes.append((v, pos[v], colour))

    texts = {}
    for u, v, txt in edges:
        if txt is not None:
            texts.setdefault((u, v), set()).add(txt)
    return {'nodes': nodes,
            'edges': [(u, v) for u, v, _txt in edges],
            'edge_labels': {uv: ', '.join(sorted(t))
                            for uv, t in sorted(texts.items())}}


def _draw_prediagram(td, ax, display_id=None):
    """Render a single typed prediagram on the provided matplotlib axis,
    as laid out by :func:`_prediagram_layout` (networkx draws it)."""
    try:
        import networkx as nx
    except ImportError:
        ax.text(0.5, 0.5, 'networkx not installed',
                ha='center', va='center')
        ax.set_axis_off()
        return

    # IMPORTANT: D is a Sage DiGraph and its vertex IDs are Sage
    # Integer types.  Passing those through to networkx → matplotlib
    # eventually leaks Sage RealLiteral into matplotlib's PDF
    # graphics-state dict, which the PDF backend can't serialize
    # ("TypeError: Don't know a PDF representation for
    # <class 'sage.rings.real_mpfr.RealLiteral'> objects").  The layout
    # holds plain Python ints / floats only, so the downstream
    # matplotlib pipeline only ever sees pure Python types.
    lay = _prediagram_layout(td, display_id)
    G_nx = nx.MultiDiGraph()
    for v, _xy, _colour in lay['nodes']:
        G_nx.add_node(v)
    for i, (u, v) in enumerate(lay['edges']):
        G_nx.add_edge(u, v, key=str(i))
    pos = {v: xy for v, xy, _colour in lay['nodes']}

    nx.draw_networkx_nodes(
        G_nx, pos, ax=ax, node_color=[c for _v, _xy, c in lay['nodes']],
        node_size=900, edgecolors='black', linewidths=1.0,
    )
    nx.draw_networkx_labels(
        G_nx, pos, ax=ax, font_size=10, font_color='white',
        font_weight='bold',
    )
    nx.draw_networkx_edges(
        G_nx, pos, ax=ax, edge_color='#444444',
        arrows=True, arrowsize=15,
        connectionstyle='arc3,rad=0.1',
    )
    if lay['edge_labels']:
        nx.draw_networkx_edge_labels(
            G_nx, pos, edge_labels=lay['edge_labels'], ax=ax, font_size=8,
            bbox={'boxstyle': 'round,pad=0.15',
                  'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 1.3)
    ax.set_axis_off()


def _draw_cover_page(pdf, model, k, max_ell, fundamental,
                     external_fields, result):
    """Page 1: model name, parameters, MF values, total slice plot."""
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.5, 2.5])

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_axis_off()
    title_lines = [
        f"MSR-JD Diagrammatic Report",
        f"Model: {model.get('name', '<unnamed>')}",
        f"k = {k},  max_ell = {max_ell},  "
        f"external_fields = {external_fields}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    ax_title.text(
        0.02, 0.5, '\n'.join(title_lines), fontsize=12, va='center',
        family='monospace',
    )

    # Parameters table
    ax_params = fig.add_subplot(gs[1, 0])
    ax_params.set_axis_off()
    ax_params.set_title('Fundamental parameters', fontsize=11, loc='left')
    params_text = []
    for key, val in fundamental.items():
        if isinstance(val, list):
            val_repr = str(val)
            if len(val_repr) > 35:
                val_repr = val_repr[:32] + '...'
        else:
            try:
                val_repr = f'{float(val):.4g}'
            except (TypeError, ValueError):
                val_repr = str(val)
        params_text.append(f'  {key:<20} = {val_repr}')
    ax_params.text(
        0.0, 1.0, '\n'.join(params_text), fontsize=9, va='top',
        family='monospace',
    )

    # MF values
    ax_mf = fig.add_subplot(gs[1, 1])
    ax_mf.set_axis_off()
    ax_mf.set_title('Mean-field solution', fontsize=11, loc='left')
    mf = result['mf_values']
    mf_lines = []
    # Classic n/v/m saddles first (their historical order), then any other
    # saddle the model declares (e.g. ``xstar``); a model need not have
    # ``nstar``/``vstar`` at all.
    legacy = ('nstar', 'vstar', 'mstar')
    names = [n for n in legacy if n in mf] + [n for n in mf if n not in legacy]
    for name in names:
        vals = mf.get(name)
        label = name[:-4] + '*' if name.endswith('star') else name
        if vals is None:
            continue
        for i, v in enumerate(vals, 1):
            mf_lines.append(f'  {label}_{i} = {v:.4f}')
    ax_mf.text(
        0.0, 1.0, '\n'.join(mf_lines), fontsize=9, va='top',
        family='monospace',
    )

    # Total slice plot (k=2).  Defensive numpy cast: any Sage RealLiteral
    # in the C_tau array would later leak into matplotlib's PDF graphics
    # state and crash PdfPages.close().
    ax_slice = fig.add_subplot(gs[2, :])
    if result['C_tau'] is not None:
        tau_grid = np.asarray(result['tau_grid'], dtype=float)
        c_arr    = np.asarray(result['C_tau'], dtype=complex)
        ax_slice.plot(tau_grid, c_arr.real.astype(float), color='#2266CC',
                      linewidth=1.6, label=r'$\mathrm{Re}\, C^{(k)}(\tau)$')
        if np.any(np.abs(c_arr.imag) > 1e-9):
            ax_slice.plot(tau_grid, c_arr.imag.astype(float),
                          color='#CC4422', linewidth=1.0, linestyle='--',
                          alpha=0.8, label=r'$\mathrm{Im}$')
        ax_slice.axhline(0, color='gray', linewidth=0.5)
        ax_slice.set_xlabel(r'$\tau$')
        ax_slice.set_ylabel(rf'$C^{{({k})}}(\tau)$')
        ax_slice.set_title(
            f'Total cumulant slice  ({len(result["diagrams"])} diagrams)',
            fontsize=11,
        )
        ax_slice.legend(loc='best', fontsize=9)
        ax_slice.grid(True, alpha=0.25)
    else:
        ax_slice.text(0.5, 0.5, f'k={k}: no slice plotted (k≥3 not yet '
                                f'evaluated on grid by compute_cumulants).',
                      ha='center', va='center', fontsize=10)
        ax_slice.set_axis_off()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _per_diagram_contributions(result):
    """Per-diagram Phase J callables, keyed by ``id(typed_diagram)``.

    Matched by object identity, never by page position.  ``compute_cumulants``
    hands each ``ell``'s typed diagrams to ``compute_correction_td`` in
    ``result['diagrams']`` order, and a per-diagram group's ``kernel_id`` is
    the index into THAT per-``ell`` list; rebuilding the list here from the
    same records maps every callable back to its own diagram whatever order
    the enumeration produced.  The grouped path (``use_grouped_phase_j``)
    has no per-diagram callables (its groups carry no ``'contribution'``),
    so its pages show none.
    """
    out = {}
    for ell, pj in (result.get('phase_j_by_ell') or {}).items():
        if not pj:
            continue
        tds = [r['typed_diagram'] for r in result.get('diagrams') or []
               if r.get('ell') == ell]
        for g in pj.get('groups') or []:
            fn, kid = g.get('contribution'), g.get('kernel_id')
            if fn is None or not isinstance(kid, int) or not 0 <= kid < len(tds):
                continue
            out[id(tds[kid])] = fn
    return out


def _pages_in_canonical_order(records):
    """``[(sort_key, display_id, record), ...]`` in page order: by loop order,
    then by isomorphism class (:func:`_canonical_view`), so "Diagram i / N"
    names the same class whatever order ``records`` came in."""
    pages = []
    for rec in records:
        key, display_id = _canonical_view(rec['typed_diagram'])
        pages.append(((int(rec.get('ell', 0)), key), display_id, rec))
    pages.sort(key=lambda page: page[0])
    return pages


def _diagram_contribution_curve(contrib, tau_grid, k):
    """One diagram's k=2 slice C_Γ(0, τ) on ``tau_grid``, sampled like the
    total (``C_tau``): Itô left limit at τ=0."""
    from api.compute import _ito_nudge_callable
    fn = _ito_nudge_callable(contrib, k)
    return np.array([complex(fn(0.0, float(t))) for t in tau_grid],
                    dtype=complex)


def _assignment_lines(td, display_id):
    """The page's "Vertex assignments" block, vertices in display order."""
    lines = ['Vertex assignments:']
    for v, vt in sorted(td.vertex_assignments.items(),
                        key=lambda item: display_id[item[0]]):
        cls = type(vt).__name__
        rl = getattr(vt, 'response_legs', [])
        pl = getattr(vt, 'physical_legs', None)
        line = f'  v{display_id[v]} ({cls}): resp={rl}'
        if pl is not None:
            line += f', phys={pl}'
        lines.append(line)
    return lines


def _draw_diagram_page(pdf, idx, total, td_record, result, k, contrib=None,
                       display_id=None):
    """One page per diagram: graph, vertex assignments, contribution.

    ``contrib`` is this diagram's own Phase J callable (from
    :func:`_per_diagram_contributions`), or None when there is none.
    ``display_id`` maps vertices to the numbers printed on the page
    (:func:`_canonical_view`); by default the canonical numbering is
    computed here."""
    td = td_record['typed_diagram']
    info = td_record['classify']
    pf = td_record['combined_prefactor']
    if display_id is None:
        display_id = _canonical_view(td)[1]

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 2.0])

    # Title row
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.set_axis_off()
    M = info.get('Scal', '?')
    title_lines = [
        f'Diagram {idx} / {total}',
        f'M = {M}',
        f'Scalar prefactor:',
        f'  {str(pf)[:80]}',
    ]
    ax_title.text(0.02, 0.95, '\n'.join(title_lines), fontsize=10,
                  va='top', family='monospace')

    # Vertex / source assignments
    ax_assign = fig.add_subplot(gs[0, 1])
    ax_assign.set_axis_off()
    ax_assign.text(0.0, 1.0, '\n'.join(_assignment_lines(td, display_id)),
                   fontsize=8, va='top', family='monospace')

    # Diagram graph
    ax_graph = fig.add_subplot(gs[1:, 0])
    _draw_prediagram(td, ax_graph, display_id)
    ax_graph.set_title('Prediagram + edge typings', fontsize=10)

    # Per-diagram contribution slice (k=2)
    ax_contrib = fig.add_subplot(gs[1:, 1])
    if k == 2:
        try:
            if contrib is not None:
                tau_grid = np.asarray(result['tau_grid'], dtype=float)
                C_diag = _diagram_contribution_curve(contrib, tau_grid, k)
                ax_contrib.plot(tau_grid, C_diag.real.astype(float),
                                color='#0066CC', linewidth=1.4)
                ax_contrib.axhline(0, color='gray', linewidth=0.5)
                ax_contrib.set_xlabel(r'$\tau$')
                ax_contrib.set_ylabel(r'$C^{(\Gamma)}_{\mathrm{Re}}$')
                ax_contrib.set_title(
                    f"This diagram's contribution",
                    fontsize=10,
                )
                ax_contrib.grid(True, alpha=0.25)
            else:
                ax_contrib.text(0.5, 0.5, '(no per-diagram callable)',
                                ha='center', va='center')
                ax_contrib.set_axis_off()
        except Exception as e:
            ax_contrib.text(0.5, 0.5, f'(plot error: {e})',
                            ha='center', va='center', fontsize=8)
            ax_contrib.set_axis_off()
    else:
        ax_contrib.text(0.5, 0.5, '(per-diagram slices for k≥3 deferred)',
                        ha='center', va='center', fontsize=10)
        ax_contrib.set_axis_off()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(
    model: dict,
    k: int,
    fundamental: dict = None,
    external_fields: list[tuple[str, int]] = None,
    output_pdf: str = None,
    *,
    parameters: dict = None,         # canonical name for ``fundamental`` (wins if both given)
    max_ell: int = 0,
    tau_max: float = 50.0,
    tau_step: float = 0.5,
    spatial_grid=None,
    taylor_order: int = None,    # auto: max(k + 2·max_ell, 2)
    use_cache: bool = True,
    verbose: bool = True,
    result: dict = None,
) -> dict:
    """
    Run compute_cumulants() and produce a multi-page PDF report.

    If ``result`` is provided (e.g., from a prior compute_cumulants()
    call), reuses it instead of recomputing.

    Returns the result dict so the caller can do further analysis.
    """
    # Accept the canonical name ``parameters`` (the dd.Config name); it wins
    # over the legacy ``fundamental`` if both are given.
    if parameters is not None:
        fundamental = parameters
    if output_pdf is None:
        raise ValueError('generate_report needs output_pdf=.')
    if result is None:
        if fundamental is None:
            raise ValueError(
                'generate_report needs parameters= (or fundamental=).')
        if external_fields is None:
            raise ValueError('generate_report needs external_fields=.')
        result = compute_cumulants(
            model           = model,
            k               = k,
            max_ell         = max_ell,
            fundamental     = fundamental,
            external_fields = external_fields,
            tau_max         = tau_max,
            tau_step        = tau_step,
            spatial_grid    = spatial_grid,
            taylor_order    = taylor_order,
            use_cache       = use_cache,
            verbose         = verbose,
        )

    # Spatial models compute C(x,τ) fine, but the multi-page PDF layout below is
    # temporal-specific (per-diagram Phase-J pages keyed on result['diagrams']).
    # Return the computed correlator without a PDF rather than KeyError'ing.
    if result.get('config', {}).get('spatial') or 'diagrams' not in result:
        if verbose:
            print('[report] spatial model: C(x,τ) computed and returned in the '
                  'result dict; the multi-page PDF report is temporal-only, so no '
                  'PDF was written.  Render spatial results in a notebook '
                  '(see notebooks/spatial/).')
        return result

    # Backstop: never let a path built from ``str(params_dict)`` create a
    # junk ``{...}/`` directory (see api.save._sanitize_output_path).
    from api.save import _sanitize_output_path
    output_pdf = _sanitize_output_path(output_pdf)

    if verbose:
        print(f'[report] writing {output_pdf} '
              f'({len(result["diagrams"])} diagram pages + cover)...')

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or '.',
                exist_ok=True)

    # ── Defensive matplotlib state reset ───────────────────────────
    # When generate_report is called from inside a Jupyter notebook
    # AFTER prior cells have produced inline figures (e.g. a
    # `plt.subplots(); plt.show()` overlay), matplotlib retains
    # figure references and a mathtext cache that can hold non-PDF-
    # serializable objects (notably Sage RealLiteral values that
    # entered through axis-label / text rendering somewhere upstream).
    # The pollution only manifests at PdfPages.close() / finalize()
    # because that's when the cumulative ExtGState dict gets dumped.
    # Closing ALL existing figures forces a clean slate for the PDF
    # we're about to write.  Standalone scripts won't notice; only
    # notebook re-runs benefit.
    plt.close('all')

    # Build the report inside a "pdf" backend rcParams scope so
    # matplotlib doesn't try to share mathtext caches with the
    # inline / agg backends used by the calling notebook.
    import matplotlib as _mpl
    with _mpl.rc_context({'text.usetex': False}):
        with PdfPages(output_pdf) as pdf:
            _draw_cover_page(pdf, model, k, max_ell, fundamental,
                             external_fields, result)
            n_diag = len(result['diagrams'])
            contribs = _per_diagram_contributions(result)
            for idx, (_key, display_id, td_record) in enumerate(
                    _pages_in_canonical_order(result['diagrams']), 1):
                _draw_diagram_page(
                    pdf, idx, n_diag, td_record, result, k,
                    contribs.get(id(td_record['typed_diagram'])),
                    display_id)
    plt.close('all')

    if verbose:
        print(f'[report] done.')
    return result

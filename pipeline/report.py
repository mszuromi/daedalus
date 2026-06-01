"""
pipeline.report — multi-page PDF showing prediagrams, typed-diagram
assignments, and per-diagram numerical contributions.

The report is meant for *intuition-building*:  scroll through diagrams,
see what the cumulant slice looks like, see how each diagram contributes
to the total.

Status (prototype):
  ✓ cover page (model name, parameters, MF values, total slice plot)
  ✓ one page per typed diagram with:
      - prediagram graph rendered via networkx + matplotlib
      - vertex assignments table
      - edge propagator labels
      - per-diagram contribution C_Γ(τ) line plot
  ✗ rich LaTeX rendering of action / vertex coefficients (TODO: pdflatex
    or matplotlib-mathtext fallback)
  ✗ symbolic propagator matrix display (deferred; the user typically
    inspects this in the notebook directly)
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pipeline.compute import compute_cumulants


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


def _draw_prediagram(td, ax):
    """Render a single typed prediagram on the provided matplotlib axis.

    Uses networkx for layout: leaves at the top (external phys), source
    / interaction vertices at the bottom.  Edge labels show the edge's
    (resp_leg, phys_leg) propagator pair.
    """
    try:
        import networkx as nx
    except ImportError:
        ax.text(0.5, 0.5, 'networkx not installed',
                ha='center', va='center')
        ax.set_axis_off()
        return

    # prediagram tuple is (D, G, leaves, internal)
    # IMPORTANT: D is a Sage DiGraph and its vertex IDs are Sage
    # Integer types.  Passing those through to networkx → matplotlib
    # eventually leaks Sage RealLiteral into matplotlib's PDF
    # graphics-state dict, which the PDF backend can't serialize
    # ("TypeError: Don't know a PDF representation for
    # <class 'sage.rings.real_mpfr.RealLiteral'> objects").  Cast every
    # vertex / edge endpoint to plain Python int up front so the
    # downstream matplotlib pipeline only ever sees pure Python types.
    D = td.prediagram[0]
    leaves_sage = list(td.prediagram[2])
    leaves      = [int(v) for v in leaves_sage]
    leaf_set    = set(leaves)
    # vertex-id mapping (sage → int) used everywhere below
    _vmap       = {v: int(v) for v in D.vertices()}

    G_nx = nx.MultiDiGraph()
    for v in D.vertices():
        G_nx.add_node(_vmap[v])
    for u, v, lbl in D.edges():
        G_nx.add_edge(_vmap[u], _vmap[v], key=str(lbl))

    # Layout: leaves at top (y=1), internal vertices at bottom (y=0)
    pos = {}
    leaf_xs = np.linspace(0.0, 1.0, max(len(leaves), 2))
    for j, lf in enumerate(leaves):
        pos[lf] = (float(leaf_xs[j]), 1.0)
    internal = [int(v) for v in D.vertices() if int(v) not in leaf_set]
    int_xs = np.linspace(0.2, 0.8, max(len(internal), 1))
    for j, v in enumerate(internal):
        pos[v] = (float(int_xs[j]), 0.0)

    # Node colors: leaves green, source vertex blue, interaction red
    node_colors = []
    for v_sage in D.vertices():
        v = _vmap[v_sage]
        if v in leaf_set:
            node_colors.append('#2ECC71')        # green = leaf
        else:
            vt = td.vertex_assignments.get(v_sage)
            if vt is None:
                node_colors.append('#999999')
            elif hasattr(vt, 'physical_legs'):
                node_colors.append('#E74C3C')    # red = interaction
            else:
                node_colors.append('#3498DB')    # blue = source

    nx.draw_networkx_nodes(
        G_nx, pos, ax=ax, node_color=node_colors,
        node_size=900, edgecolors='black', linewidths=1.0,
    )
    nx.draw_networkx_labels(
        G_nx, pos, ax=ax, font_size=10, font_color='white',
        font_weight='bold',
    )
    # Draw edges with labels — keys are Python (int, int) tuples
    edge_labels = {}
    for u, v, lbl in D.edges():
        et = td.edge_types.get((u, v, lbl))
        if et is not None:
            resp_leg, phys_leg = et
            edge_labels[(_vmap[u], _vmap[v])] = (
                rf'${resp_leg[0]}_{int(resp_leg[1])}\!\to\!'
                rf'\,{phys_leg[0]}_{int(phys_leg[1])}$'
            )
    nx.draw_networkx_edges(
        G_nx, pos, ax=ax, edge_color='#444444',
        arrows=True, arrowsize=15,
        connectionstyle='arc3,rad=0.1',
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G_nx, pos, edge_labels=edge_labels, ax=ax, font_size=8,
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
    for label, vals in [('n*', mf['nstar']), ('v*', mf['vstar']),
                        ('m*', mf.get('mstar'))]:
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


def _draw_diagram_page(pdf, idx, total, td_record, result, k):
    """One page per diagram: graph, vertex assignments, contribution."""
    td = td_record['typed_diagram']
    info = td_record['classify']
    pf = td_record['combined_prefactor']

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 2.0])

    # Title row
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.set_axis_off()
    M = info.get('M', '?')
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
    assign_lines = ['Vertex assignments:']
    for v, vt in sorted(td.vertex_assignments.items()):
        cls = type(vt).__name__
        rl = getattr(vt, 'response_legs', [])
        pl = getattr(vt, 'physical_legs', None)
        line = f'  v{v} ({cls}): resp={rl}'
        if pl is not None:
            line += f', phys={pl}'
        assign_lines.append(line)
    ax_assign.text(0.0, 1.0, '\n'.join(assign_lines), fontsize=8,
                   va='top', family='monospace')

    # Diagram graph
    ax_graph = fig.add_subplot(gs[1:, 0])
    _draw_prediagram(td, ax_graph)
    ax_graph.set_title('Prediagram + edge typings', fontsize=10)

    # Per-diagram contribution slice (k=2)
    ax_contrib = fig.add_subplot(gs[1:, 1])
    if k == 2:
        try:
            phase_j = result['phase_j_result']
            # The compute_correction_td result has per-diagram callables
            # in 'tree_callables' (list, indexed by kernel-group order)
            tree_callables = phase_j.get('tree_callables', [])
            if idx - 1 < len(tree_callables):
                contrib = tree_callables[idx - 1]
                tau_grid = np.asarray(result['tau_grid'], dtype=float)
                C_diag = np.array([
                    complex(contrib(0.0, float(t))) for t in tau_grid
                ], dtype=complex)
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
    fundamental: dict,
    external_fields: list[tuple[str, int]],
    output_pdf: str,
    *,
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
    if result is None:
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
    # junk ``{...}/`` directory (see pipeline.save._sanitize_output_path).
    from pipeline.save import _sanitize_output_path
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
            for idx, td_record in enumerate(result['diagrams'], 1):
                _draw_diagram_page(pdf, idx, n_diag, td_record,
                                   result, k)
    plt.close('all')

    if verbose:
        print(f'[report] done.')
    return result

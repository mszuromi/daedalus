"""
pipeline.ui.main — the TheoryUI form.

Composes the 9-section tab editor that ``notebooks/theory_builder.ipynb``
launches, with a Save button that writes a ``.theory.py`` file via
``pipeline.theory_serialize``.

Usage in a notebook
-------------------
::

    from pipeline.ui import TheoryUI
    ui = TheoryUI()
    ui.show()

    # … fill out the form, hit Save …

    # The saved spec stays accessible:
    spec = ui.spec()       # the form-state dict
    path = ui.last_saved   # path of the last save
"""
from __future__ import annotations

import os
from typing import Any, Optional

import ipywidgets as W
from IPython.display import display, HTML

from pipeline.ui.widgets import (
    DynamicTable,
    expression_input,
    matrix_input,
    textarea_input,
    vector_input,
)
from pipeline.theory_serialize import (
    save_theory_to_file,
    render_theory_file,
    load_spec_from_file,
)


# Default theories directory — relative to current working dir, since
# notebooks/ runs from there
_DEFAULT_THEORIES_DIR = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'theories')
)


# ── CSS overhaul (2026-05-26) ─────────────────────────────────────────
# One inline stylesheet, injected at ``show()`` time.  Targets the
# specific CSS classes we add via ``add_class()`` on the relevant
# ipywidgets primitives (``tb-…`` namespace to avoid colliding with
# Jupyter / Lab themes).  Design tokens:
#   • palette: blue accent (#3b82f6), neutral border (#e5e7eb),
#              text (#111827), muted text (#6b7280), card bg (#fafafa)
#   • spacing scale (4-unit): 4 / 8 / 12 / 16 / 24
#   • code font: ui-monospace (system mono) → JetBrains Mono → Menlo
#   • 4-button palette: btn-primary / btn-secondary / btn-link /
#                       btn-muted (replaces ad-hoc button_style mixes)
_THEORY_BUILDER_CSS = """
<style>
/* ── Outer container ──────────────────────────────────────────── */
.tb-root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: #111827;
}
.tb-root h2 { color: #111827; font-weight: 600; }
.tb-root h4 {
    margin: 0 0 8px 0; color: #111827; font-weight: 600;
    font-size: 1.05em;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 6px;
}
.tb-root p, .tb-root li {
    color: #4b5563; font-size: 90%; line-height: 1.55;
}
.tb-root code {
    background: #f3f4f6; padding: 1px 5px; border-radius: 3px;
    font-family: ui-monospace, "JetBrains Mono", Menlo, "SF Mono",
                 Consolas, monospace;
    font-size: 92%;
}
.tb-root pre {
    background: #f3f4f6; padding: 8px 12px; border-radius: 4px;
    border-left: 3px solid #3b82f6;
    font-family: ui-monospace, "JetBrains Mono", Menlo, "SF Mono",
                 Consolas, monospace;
    font-size: 0.9em; line-height: 1.5;
    overflow-x: auto;
}

/* ── Tab strip ────────────────────────────────────────────────── */
.tb-root .widget-tab > .p-TabBar .p-TabBar-tab {
    background: transparent;
    border-bottom: 2px solid transparent;
    font-weight: 500;
    color: #6b7280;
    transition: color 0.15s, border-color 0.15s;
}
.tb-root .widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current {
    border-bottom: 2px solid #3b82f6;
    color: #111827;
    font-weight: 600;
    background: transparent;
}
.tb-root .widget-tab > .p-TabBar .p-TabBar-tab:hover {
    color: #111827;
}

/* ── Each tab content panel = a card ──────────────────────────── */
.tb-tab-panel {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 16px 20px !important;
    margin-top: 4px;
}

/* ── Dynamic-table row striping & header ──────────────────────── */
.tb-table-header > div, .tb-table-header b {
    color: #374151 !important;
    font-size: 88% !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
}
.tb-table-body > .widget-hbox:nth-child(even) {
    background: #f9fafb;
}
.tb-table-body > .widget-hbox {
    padding: 2px 4px;
    border-radius: 3px;
}

/* ── Action / MF / seed-box textareas — code feel ─────────────── */
.tb-code-area textarea {
    font-family: ui-monospace, "JetBrains Mono", Menlo, "SF Mono",
                 Consolas, monospace !important;
    font-size: 0.92em !important;
    line-height: 1.55 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 4px !important;
    padding: 8px !important;
    background: #fcfcfd !important;
}
.tb-code-area textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
}

/* ── Four-button palette ──────────────────────────────────────── */
.tb-root .tb-btn-primary button {
    background: #3b82f6 !important; color: white !important;
    border: 1px solid #2563eb !important;
    font-weight: 600 !important;
}
.tb-root .tb-btn-primary button:hover {
    background: #2563eb !important;
}
.tb-root .tb-btn-secondary button {
    background: white !important; color: #374151 !important;
    border: 1px solid #d1d5db !important;
    font-weight: 500 !important;
}
.tb-root .tb-btn-secondary button:hover {
    background: #f3f4f6 !important;
    border-color: #9ca3af !important;
}
.tb-root .tb-btn-link button {
    background: transparent !important; color: #2563eb !important;
    border: 1px solid transparent !important;
    font-weight: 500 !important;
}
.tb-root .tb-btn-link button:hover {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
}
.tb-root .tb-btn-muted button {
    background: #f9fafb !important; color: #6b7280 !important;
    border: 1px solid #e5e7eb !important;
    font-weight: 500 !important;
}
.tb-root .tb-btn-muted button:hover {
    background: #f3f4f6 !important; color: #374151 !important;
}
.tb-root .tb-btn-danger button {
    background: white !important; color: #b91c1c !important;
    border: 1px solid #fca5a5 !important;
    font-weight: 500 !important;
}
.tb-root .tb-btn-danger button:hover {
    background: #fef2f2 !important;
}

/* ── Validation sidebar ──────────────────────────────────────── */
.tb-validation {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0 12px 0;
    font-size: 90%;
    color: #374151;
}
.tb-validation b { color: #111827; }
.tb-validation .tb-v-ok    { color: #15803d; }
.tb-validation .tb-v-warn  { color: #b45309; }
.tb-validation .tb-v-error { color: #b91c1c; }
.tb-validation ul { margin: 4px 0 0 16px; padding-left: 0; }
.tb-validation li { margin: 2px 0; }

/* ── Status panel ────────────────────────────────────────────── */
.tb-status {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
    font-family: ui-monospace, "JetBrains Mono", Menlo, "SF Mono",
                 Consolas, monospace !important;
    font-size: 0.88em !important;
}

/* ── Save-path field — show derived filename next to it ──────── */
.tb-save-hint {
    color: #6b7280; font-size: 88%; padding-left: 8px;
    font-family: ui-monospace, "JetBrains Mono", Menlo, monospace;
}

/* ── Header banner ───────────────────────────────────────────── */
.tb-header {
    border-left: 4px solid #3b82f6;
    padding: 4px 0 4px 14px;
    margin-bottom: 12px;
}
.tb-header h2 { margin: 0 0 4px 0; }
.tb-header p  { margin: 0; }
</style>
"""
_CSS_INJECTED = False  # module-level guard — inject the <style> once
                       # per kernel session (multiple TheoryUI() calls
                       # would otherwise repeat the stylesheet).


def _normalize_cgf_rows(rows: list[dict]) -> list[dict]:
    """Convert raw UI CGF rows into spec-form CGF rows.

    UI rows carry ``response_legs`` as a comma-separated text string.
    The spec dict shape uses either:
      * ``response_legs`` (list of strings) for multi-leg cumulants;
      * ``response_field`` (single string) for the legacy single-leg
        case (also kept on multi-leg rows as a back-compat hint = the
        first leg's name, so older loaders that only know about
        ``response_field`` still get something sensible).
    """
    out: list[dict] = []
    for r in rows:
        legs_text = (r.get('response_legs') or '').strip()
        if not legs_text:
            # Empty row — leave it untouched; the row-filter downstream
            # (which checks lhs/coefficient/etc.) will drop it if blank.
            out.append(dict(r))
            continue
        parts = [s.strip() for s in legs_text.split(',') if s.strip()]
        if len(parts) <= 1:
            # Single-leg row — emit the legacy single-field key for
            # back-compat with the old loader / older theory files.
            out.append({
                **{k: v for k, v in r.items() if k != 'response_legs'},
                'response_field': parts[0] if parts else '',
                'response_legs':  None,
            })
        else:
            # Multi-leg cross-field row.  Keep response_legs as a list;
            # also stamp response_field with the first leg's name as a
            # cosmetic fallback for older readers.
            out.append({
                **{k: v for k, v in r.items() if k != 'response_legs'},
                'response_field': parts[0],
                'response_legs':  parts,
            })
    return out


class TheoryUI:
    """Notebook-based form for declaring stochastic theories.

    Render with :meth:`show`.  After saving, :attr:`last_saved` holds
    the path written and :meth:`spec` returns the form's spec dict.
    """

    def __init__(self, theories_dir: Optional[str] = None):
        self.theories_dir = (theories_dir
                             or _DEFAULT_THEORIES_DIR)
        os.makedirs(self.theories_dir, exist_ok=True)
        self.last_saved: Optional[str] = None
        # Unsaved-changes tracking — flipped True by ``_mark_changed``
        # whenever any widget fires its observer, reset by Save / Load
        # / Reset.  Gates Reset and Load behind a confirm checkbox in
        # the bottom bar to prevent accidental loss of work.
        self._dirty: bool = False
        # Carry-through for spec keys that no longer have a UI input.
        # ``latex`` for fields / functions / kernels used to be a UI
        # column; the column was removed (2026-05-27) but old theory
        # files may still carry custom strings.  Populated by
        # ``load()`` and re-injected by ``_collect()`` so a load /
        # edit / save round-trip preserves them.  Keyed by name to
        # survive renames upstream that re-order rows.
        self._loaded_extras: dict[str, dict[str, str]] = {
            'field_latex':    {},
            'function_latex': {},
            'kernel_latex':   {},
        }
        self._build_widgets()

    # ── Population helpers ────────────────────────────────────────
    def _pop_names(self) -> list[str]:
        """Current list of population names from the Populations tab.
        Used as a dynamic ``options_provider`` for population-aware
        dropdowns on other tabs."""
        if not hasattr(self, '_tbl_populations'):
            return []
        return [r['name'] for r in self._tbl_populations.get_rows()
                if (r.get('name') or '').strip()]

    def _pop_size_map(self) -> dict[str, int]:
        """{population_name: size} dict from the current Populations tab."""
        out = {}
        if not hasattr(self, '_tbl_populations'):
            return out
        for r in self._tbl_populations.get_rows():
            name = (r.get('name') or '').strip()
            try:
                size = int(r.get('size') or 0)
            except (TypeError, ValueError):
                size = 0
            if name and size > 0:
                out[name] = size
        return out

    def _autofill_default_templates(self) -> None:
        """Walk every Parameters row; if its ``default`` cell is empty
        and ``index_1`` / ``index_2`` describe a shape, fill in a
        template like ``[, , ]`` (vector) or ``[[, , ], [, , ]]``
        (matrix).  Preserves any non-empty user-typed values."""
        if not hasattr(self, '_tbl_parameters'):
            return
        sizes = self._pop_size_map()
        _NONE = '—'

        def _vector_template(n: int) -> str:
            return '[' + ', '.join([''] * n) + ']'

        def _matrix_template(n_rows: int, n_cols: int) -> str:
            row = _vector_template(n_cols)
            return '[' + ', '.join([row] * n_rows) + ']'

        for w_dict in self._tbl_parameters._row_widgets:
            i1 = w_dict.get('index_1')
            i2 = w_dict.get('index_2')
            d  = w_dict.get('default')
            if i1 is None or d is None:
                continue
            cur = (d.value or '').strip()
            if cur:                       # user has typed something — don't overwrite
                continue
            n1 = sizes.get(i1.value if i1.value != _NONE else None)
            n2 = sizes.get(i2.value if (i2 is not None
                                        and i2.value != _NONE) else None)
            if n1 and n2:
                d.value = _matrix_template(n1, n2)
            elif n1:
                d.value = _vector_template(n1)
            # both blank → scalar; leave empty for user to type

    # ── Tab construction ──────────────────────────────────────────
    def _build_widgets(self) -> None:
        # Tab 1: Model — name + description
        # ``n_populations`` is derived from the Populations tab; no
        # standalone spinner on this tab.
        self._w_name = W.Text(
            value='My Stochastic Theory', placeholder='Theory name',
            layout=W.Layout(width='400px'),
            description='Name:', style={'description_width': '80px'},
        )
        self._w_description = W.Textarea(
            value='', placeholder='Optional theory description / notes',
            layout=W.Layout(width='600px', height='60px'),
            description='Description:',
            style={'description_width': '120px'},
        )
        tab_model = W.VBox([
            W.HTML(
                '<h4>Theory metadata</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Give your theory a name (used to derive the "
                "<code>.theory.py</code> filename on save) and an "
                "optional one-line description.  Everything else "
                "&mdash; fields, parameters, dynamics &mdash; lives on "
                "the tabs to the right."
                '</p>'),
            self._w_name, self._w_description,
        ])

        # Tab 2: Populations — declare named populations + their sizes.
        # Other tabs read the current population list to populate
        # dropdowns: a field belongs to one population; a parameter /
        # kernel is indexed by zero / one / two populations.
        self._tbl_populations = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'A',    'width': '120px'},
                {'name': 'size',        'kind': 'int',
                 'default': 1,          'width': '80px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': '(optional) description of this population',
                 'width': '320px'},
            ],
            # No starter rows.  A scalar / single-field theory needs no
            # populations at all; surfacing two pre-filled rows ("pop1",
            # "pop2") tricks the user into thinking they're required and
            # — worse — leaves the unused defaults in the saved spec.
            # Click "+ add row" to declare populations only when needed.
            initial=[],
        )
        tab_populations = W.VBox([
            W.HTML(
                '<h4>Populations</h4>'
                '<p style="color:#555;font-size:90%;">'
                "<b>Skip this tab</b> if your theory has just a single "
                "scalar field per variable (most 1D and 2D Langevin "
                "examples).  Populations are only needed when you have "
                "groups of <i>multiple</i> identical units that share "
                "the same dynamics &mdash; e.g. <i>N</i> neurons in a "
                "network, <i>N</i> spins in a lattice."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "When you do need them: each row declares one named "
                "group.  The <b>name</b> is any identifier (<code>A</code>, "
                "<code>E</code>, <code>spins</code>, &hellip;); the <b>size</b> is "
                "the positive integer count of units in that group.  "
                "Once a population is declared, you can attach fields, "
                "parameters, and kernels to it on the later tabs."
                '</p>'),
            self._tbl_populations.show(),
        ])

        # Tab 3: Physical fields.
        # Each field belongs to exactly ONE population (declared in
        # the Populations tab).  The framework auto-creates:
        #   - fluctuation:  d<name>     (used in the action)
        #   - response:     <name>t     (MSR-pairing factor)
        #   - saddle:       <name>star  (solved by the saddle solver)
        self._tbl_physical = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'x',    'width': '120px'},
                {'name': 'population',  'kind': 'select',
                 'options_provider': self._pop_names,
                 'width': '120px'},
                # latex column dropped (2026-05-27): the framework
                # auto-derives a sensible latex name from ``name``; the
                # column was clutter for the 95% case.  Theory files
                # that already carry a custom ``latex`` round-trip
                # through ``theory_serialize`` untouched.
                {'name': 'description', 'kind': 'text',
                 'placeholder': '(optional)', 'width': '280px'},
            ],
            initial=[
                # One starter row only — a scalar field ``x`` with no
                # population.  Generic enough to keep without nudging
                # users toward a Hawkes-style two-field setup.
                {'name': 'x', 'population': '', 'description': ''},
            ],
        )
        tab_fields = W.VBox([
            W.HTML(
                '<h4>Fields</h4>'
                '<p style="color:#555;font-size:90%;">'
                "These are the actual physical quantities your theory "
                "describes &mdash; the variables you'd write in an SDE.  "
                "Give each one a short name (<code>x</code>, <code>y</code>, "
                "<code>v</code>, &hellip;) and the population it belongs to "
                "(leave <b>blank</b> for a scalar field with no population)."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "For every field <code>x</code> you declare, the "
                "framework automatically creates two companion symbols "
                "you'll need when writing the action and the MF equations:"
                '<ul style="margin-top:2px;">'
                '<li><code>xt</code> &mdash; the <i>response</i> field '
                '(MSR auxiliary).  Appears in the action paired with '
                'the dynamics: <code>xt * ((Dt + mu) * x + &hellip;)</code>.</li>'
                '<li><code>xstar</code> &mdash; the <i>saddle</i> '
                '(steady-state value).  Solved numerically from the '
                'mean-field equations.</li>'
                '</ul>'
                "If you attach a population <code>A</code>, all three "
                "(<code>x</code>, <code>xt</code>, <code>xstar</code>) become "
                "vectors of length <i>size(A)</i>, and the action / MF "
                "equations need a <code>for i in A</code> comprehension "
                "to iterate over them."
                '</p>'),
            self._tbl_physical.show(),
        ])

        # Tab 4: Parameters.
        # ``index_1`` / ``index_2`` are dropdowns over the declared
        # populations (or '—' for none).  Both empty → scalar.
        # One filled → vector of length size(pop).  Both filled →
        # matrix of shape (size(pop_1), size(pop_2)) — row-first.
        # When the user changes either index dropdown, the ``default``
        # placeholder auto-templates to the right shape (e.g.
        # ``[[, , ], [, , ]]`` for a 2×3 matrix).
        _NONE = '—'
        def _pop_opts_with_none():
            return [_NONE] + self._pop_names()
        self._tbl_parameters = DynamicTable(
            columns=[
                {'name': 'name',     'kind': 'text',
                 'placeholder': 'mu', 'width': '110px'},
                {'name': 'index_1',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '100px'},
                {'name': 'index_2',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '100px'},
                {'name': 'domain',   'kind': 'select',
                 'options': ['', 'positive', 'real'],
                 'default': '', 'width': '90px'},
                {'name': 'default',  'kind': 'text',
                 'placeholder': '1.0  /  [.., ..]  /  [[..]]',
                 'width': '300px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': '(optional)', 'width': '180px'},
            ],
            initial=[
                # Generic scalar parameter ``mu`` (drift / linear-restoring
                # coefficient) with no specific physical interpretation.
                {'name': 'mu', 'index_1': _NONE, 'index_2': _NONE,
                 'domain': 'real', 'default': '1.0'},
            ],
        )
        tab_params = W.VBox([
            W.HTML(
                '<h4>Parameters</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Any numerical knob your action / MF equations refer to "
                "(drift coefficients, coupling strengths, noise "
                "amplitudes, &hellip;).  Each row is one parameter."
                '</p>'
                '<p style="color:#555;font-size:90%;"><b>Shape</b>: pick how the '
                "parameter is indexed using the two dropdowns:"
                '<ul style="margin-top:2px;">'
                '<li><b>Both blank</b> &rarr; <i>scalar</i>.  '
                'E.g. <code>mu = 1.0</code> in <code>dx/dt = -mu*x</code>.</li>'
                '<li><b>Only index_1</b> &rarr; <i>vector</i> of length '
                '<i>size(index_1)</i>.  Refer to it as '
                '<code>mu[i]</code> in the action.</li>'
                '<li><b>Both</b> &rarr; <i>matrix</i> of shape '
                '<i>size(index_1)</i> &times; <i>size(index_2)</i>.  '
                'Refer to it as <code>w[i, j]</code> in the action.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;">'
                "<b>domain</b>: hint to the mean-field solver about "
                "where to look for roots.  <code>positive</code> = "
                "Newton starts in <code>[0, 5&middot;scale]</code>, "
                "<code>real</code> = <code>[&minus;3&middot;scale, "
                "3&middot;scale]</code>, blank = no preference.  Affects "
                "starting guesses, not which values you can later pass "
                "at run time."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "<b>default</b>: the value the runner picks up if you "
                "don't override it.  Match the shape: a scalar parameter "
                "wants a single number, a vector parameter wants a list "
                "like <code>[1.0, 2.0]</code>, a matrix wants a list of "
                "lists like <code>[[1.0, 0.5], [0.5, 1.0]]</code>.  The "
                "placeholder updates automatically when you change the "
                "index dropdowns."
                '</p>'),
            self._tbl_parameters.show(),
        ])

        # Tab 5: Functions
        # ``population`` (single-select dropdown of declared
        # populations) tells the framework which population this
        # function is bound to.  In the action, ``phi[i](v[i])`` then
        # iterates ``i`` over that population's pop_<name> range.
        # Multi-arg functions (e.g. ``f(v, n)``) still take their
        # arg names from the comma-separated ``args`` field; if both
        # args live in the same population, just one dropdown is needed.
        self._tbl_functions = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'f',    'width': '100px'},
                {'name': 'population',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'args',        'kind': 'text',
                 'placeholder': 'x  (or "x, y" for multi-arg)',
                 'width': '140px'},
                {'name': 'expression',  'kind': 'text',
                 'placeholder': 'x^2  /  a*x + b  /  ...',
                 'width': '260px'},
                # latex column dropped (2026-05-27).
                {'name': 'description', 'kind': 'text',
                 'placeholder': '(optional)', 'width': '180px'},
            ],
            # No starter row — most theories declare zero or one
            # function, and the default ``phi(v) = a*v`` was leading users
            # to think a transfer-function declaration is mandatory.
            # Click "+ add row" to declare functions when actually needed.
            initial=[],
        )
        tab_functions = W.VBox([
            W.HTML(
                '<h4>Functions</h4>'
                '<p style="color:#555;font-size:90%;">'
                "<b>Skip this tab</b> unless your dynamics include a "
                "non-polynomial transformation of a field &mdash; e.g. "
                "<code>tanh(x)</code>, <code>exp(v)</code>, or any "
                "user-defined transfer function.  Polynomial nonlinearities "
                "(like <code>x^3</code>) go straight in the action."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "When you do need it: declare each function as a named "
                "row.  Then call it from the action with square-bracket "
                "indexing if it's population-scoped, parentheses for the "
                "argument:"
                '<ul style="margin-top:2px;">'
                '<li>Population-scoped: <code>f[i](x[i])</code>, where '
                '<code>i</code> ranges over the function\'s population.</li>'
                '<li>Global (no population): <code>f(x)</code>.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;"><b>Per row:</b><ul>'
                '<li><b>name</b> &mdash; any identifier '
                '(<code>f</code>, <code>g</code>, <code>phi</code>, &hellip;).</li>'
                '<li><b>population</b> &mdash; the population this function '
                'is indexed by, or blank for a global function.</li>'
                '<li><b>args</b> &mdash; comma-separated argument names '
                '(<code>x</code> for a 1-arg function; <code>x, y</code> '
                'for a 2-arg one).  These are just placeholder names '
                'used inside the expression.</li>'
                '<li><b>expression</b> &mdash; the function body in terms '
                'of its <code>args</code> and any declared parameters.  '
                'E.g. <code>tanh(a*x)</code>, <code>x^2 + b</code>.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;">'
                "<i>Behind the scenes</i>: the framework expands the "
                "function around the saddle value of its first argument "
                "(so <code>f(x)</code> gets a Taylor expansion in "
                "<code>x &minus; xstar</code>).  You don't need to do "
                "this expansion yourself &mdash; just write "
                "<code>f[i](x[i])</code> in the action and the framework "
                "handles the rest."
                '</p>'),
            self._tbl_functions.show(),
        ])

        # Tab 6: Kernels
        # Same index_1 / index_2 pattern as Parameters.
        self._tbl_kernels = DynamicTable(
            columns=[
                {'name': 'name',       'kind': 'text',
                 'placeholder': 'K',   'width': '100px'},
                {'name': 'index_1',    'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'index_2',    'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'time_expr',  'kind': 'text',
                 'placeholder': '(1/tauk)*exp(-t/tauk)*heaviside(t)',
                 'width': '280px'},
                {'name': 'freq_image', 'kind': 'text',
                 'placeholder': '1/(1+I*omega*tauk)',
                 'width': '220px'},
                # latex_name column dropped (2026-05-27).
            ],
            # No starter row — kernels are an optional feature (only
            # used by conv-style synapses / shaped filters), not a
            # required ingredient of every theory.
            initial=[],
        )
        tab_kernels = W.VBox([
            W.HTML(
                '<h4>Kernels</h4>'
                '<p style="color:#555;font-size:90%;">'
                "<b>Skip this tab</b> unless your action has a "
                "<i>convolution</i> term &mdash; i.e. one field "
                "filtered through a memory kernel before coupling to "
                "another, like <code>K * y</code> where "
                "<code>(K * y)(t) = &int; K(t&minus;t') y(t') dt'</code>.  "
                "Instantaneous (non-memory) couplings need no kernel."
                '</p>'
                '<p style="color:#555;font-size:90%;"><b>Per row:</b><ul>'
                '<li><b>name</b> &mdash; an identifier '
                '(<code>K</code>, <code>g</code>, &hellip;).</li>'
                '<li><b>index_1 / index_2</b> &mdash; same shape rules as '
                'Parameters: blank for a scalar kernel, one set for a '
                'per-population kernel <code>K[i]</code>, both set for '
                'a per-pair kernel <code>K[i, j]</code>.</li>'
                '<li><b>time_expr</b> &mdash; the kernel as a function of '
                '<code>t</code> (and parameters).  Use <code>heaviside(t)</code> '
                'to enforce causality.</li>'
                '<li><b>freq_image</b> &mdash; the Fourier image of the '
                'kernel: a function of <code>omega</code> (and parameters).  '
                '<b>Preferred</b> &mdash; the propagator builder uses this '
                'directly.  Fill either column; both work, but '
                '<code>freq_image</code> is faster.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;"><b>Examples</b> '
                'of a single-exponential memory kernel:'
                '<pre style="background:#f3f4f6;padding:8px;'
                'border-radius:4px;font-size:0.9em;">'
                "name=K   time_expr=(1/tauk)*exp(-t/tauk)*heaviside(t)\n"
                "         freq_image=1/(1 + I*omega*tauk)"
                '</pre></p>'
                '<p style="color:#555;font-size:90%;">'
                "Then in the action: <code>xt * Conv(K, y)</code> couples "
                "<i>x</i> to the kernel-filtered version of <i>y</i>."
                '</p>'),
            self._tbl_kernels.show(),
        ])

        # Tab 7: Noise (correlated / colored / cross-correlated noise
        # cumulants).  Internally this is the "CGF" tab — the rows feed
        # the cumulant generating functional injection
        # ``-W[mt]`` ⊃ -(1/n!) ∫ &kappa;<sup>(n)</sup> &times; mt&middot;mt&hellip;
        # — but the user-facing label is "Noise" because that's what
        # everyone calls it informally and "CGF" was unhelpful jargon.
        self._tbl_cgfs = DynamicTable(
            columns=[
                {'name': 'name',           'kind': 'text',
                 'placeholder': 'Cxx',     'width': '60px'},
                {'name': 'response_legs',  'kind': 'text',
                 'placeholder': 'xt   or   xt, yt',
                 'width': '160px'},
                {'name': 'order',          'kind': 'int',
                 'default': 2,             'width': '60px'},
                {'name': 'coefficient',    'kind': 'text',
                 'placeholder': '2*D   or   D/tauc',
                 'width': '300px'},
                {'name': 'kernel',         'kind': 'text',
                 'placeholder': 'dirac_delta(tau)   or   exp(-abs(tau)/tauc)',
                 'width': '220px'},
            ],
            initial=[],
        )
        tab_cgfs = W.VBox([
            W.HTML(
                '<h4>Noise</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Declare the random forcing that drives your theory.  "
                "Each row specifies one piece of the noise correlator "
                "<code>&lt;&eta;(t) &eta;(t')&gt; = coefficient &times; "
                "kernel(t&minus;t')</code>.  Leave the table empty if "
                "your action already includes the noise term as an "
                "explicit <code>D*xt^2</code>-style line."
                '</p>'
                '<p style="color:#555;font-size:90%;"><b>Per row:</b><ul>'
                '<li><b>name</b> &mdash; any label you choose '
                '(e.g. <code>Cxx</code>, <code>Cxy</code>).  Just a tag.</li>'
                '<li><b>response_legs</b> &mdash; which response fields '
                'this noise term couples (with the <code>t</code> suffix). '
                'Comma-separate for multi-leg correlators.<br>'
                '&nbsp;&nbsp;<code>xt</code> &nbsp;&rarr;&nbsp; '
                '<code>&lt;&eta;<sub>x</sub> &eta;<sub>x</sub>&gt;</code> '
                '(auto-correlation of <i>x</i>)<br>'
                '&nbsp;&nbsp;<code>xt, yt</code> &nbsp;&rarr;&nbsp; '
                '<code>&lt;&eta;<sub>x</sub> &eta;<sub>y</sub>&gt;</code> '
                '(cross-correlation between <i>x</i> and <i>y</i>)</li>'
                '<li><b>order</b> &mdash; the number of legs.  '
                '<code>2</code> for ordinary Gaussian noise; '
                '<code>3</code>+ for shot-noise &kappa;<sup>(n)</sup> '
                'and similar.</li>'
                '<li><b>coefficient</b> &mdash; the &tau;-independent '
                'amplitude.  E.g. <code>2*D</code> for white Gaussian, '
                '<code>D/tauc</code> for OU-colored, '
                '<code>rho*sqrt(D1*D2)/tauc</code> for cross-correlated.</li>'
                '<li><b>kernel</b> &mdash; the &tau;-dependent shape, '
                'where <code>tau = t - t\'</code>.  '
                '<code>dirac_delta(tau)</code> for white, '
                '<code>exp(-abs(tau)/tauc)</code> for OU-colored, '
                'anything you write that integrates to 1.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;">'
                '<b>Concrete examples</b> for <code>dx/dt = -mu*x + &eta;</code>:'
                '<pre style="background:#f3f4f6;padding:8px;'
                'border-radius:4px;font-size:0.9em;">'
                "# White Gaussian, <&eta; &eta;> = 2D &middot; &delta;(&tau;)\n"
                "name=Cxx   legs=xt   order=2   coeff=2*D       kernel=dirac_delta(tau)\n"
                "\n"
                "# OU-colored, <&eta; &eta;> = (D/&tau;<sub>c</sub>) exp(-|&tau;|/&tau;<sub>c</sub>)\n"
                "name=Cxx   legs=xt   order=2   coeff=D/tauc    kernel=exp(-abs(tau)/tauc)\n"
                "\n"
                "# 2D cross-correlated, <&eta;<sub>x</sub> &eta;<sub>y</sub>> = &rho;&radic;(D&#8321;D&#8322;) &middot; &delta;(&tau;)\n"
                "name=Cxy   legs=xt, yt   order=2   coeff=rho*sqrt(D1*D2)   kernel=dirac_delta(tau)"
                '</pre></p>'),
            self._tbl_cgfs.show(),
        ])

        # Tab 7: Action — full S in physical fields, with explicit sums.
        # Placeholder shows the smallest example that illustrates EVERY
        # syntactic building block at once: response × deterministic
        # dynamics, a polynomial nonlinearity, Dt as a time-derivative
        # operator, and a Gaussian-noise term written via the squared
        # response field.  Deliberately scalar and domain-agnostic so
        # the user can transpose it to any physical setup.
        self._w_action = textarea_input(
            'Action S',
            placeholder=(
                "# Example: 1D overdamped Langevin with a cubic well\n"
                "#   dx/dt = -mu*x - eps*x^3 + noise,  <eta eta> = 2D\n"
                "xt * ((Dt + mu) * x + eps * x^3) - D * xt^2\n"
                "\n"
                "# (For population-indexed theories use comprehensions:\n"
                "#   sum(xt[i]*(Dt + mu[i])*x[i] - D[i]*xt[i]^2 for i in A))"
            ),
            rows=10,
            width='820px',
        )
        tab_action = W.VBox([
            W.HTML(
                '<h4>Action</h4>'
                '<p style="color:#555;font-size:90%;">'
                "The MSR-JD action <i>S</i> &mdash; one expression that "
                "captures every term in your theory's dynamics.  The "
                "general shape is <b>response field &times; (equation "
                "of motion)</b>:"
                '<pre style="background:#f3f4f6;padding:8px;'
                'border-radius:4px;font-size:0.9em;">'
                "xt * ((Dt + mu) * x + eps * x^3) - D * xt^2\n"
                "&nbsp; &uarr;_________________________&uarr; &nbsp; &uarr;______&uarr;\n"
                "  bilinear &times; EOM                 noise term"
                '</pre></p>'
                '<p style="color:#555;font-size:90%;"><b>Conventions:</b><ul>'
                '<li><b>Physical fields go in unmodified.</b>  Write '
                '<code>x</code>, <code>v</code>, &hellip; (not '
                '<code>xstar + dx</code>).  The framework expands them '
                'into saddle + fluctuation under the hood.</li>'
                '<li><b><code>Dt</code> is the time derivative.</b>  Compose '
                'it like any operator: <code>(Dt + mu) * x</code>, '
                '<code>(tau * Dt + 1) * x</code>.</li>'
                '<li><b>Custom functions</b> declared on the Functions tab '
                'work like normal Python calls: <code>f[i](x[i])</code> for '
                "an indexed function, <code>f(x)</code> for a global one."
                '</li>'
                '<li><b>Matrix entries</b>: use either '
                '<code>w[i, j]</code> or <code>w[i][j]</code> &mdash; both '
                'parse equivalently.</li>'
                '</ul></p>'
                '<p style="color:#555;font-size:90%;"><b>Indexed (population) '
                'theories</b>: wrap each term in a Python comprehension over '
                'the population.  E.g. for a population named <code>A</code>:'
                '<pre style="background:#f3f4f6;padding:8px;'
                'border-radius:4px;font-size:0.9em;">'
                "sum(xt[i] * ((Dt + mu[i]) * x[i] + eps * x[i]^3) - D[i] * xt[i]^2\n"
                "    for i in A)"
                '</pre>'
                "Nested comprehensions for matrix couplings: "
                "<code>sum(w[i,j]*x[j] for j in A)</code>."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "<b>Noise terms</b> can go in the action directly "
                "(<code>&minus;D*xt^2</code> for white Gaussian) OR be "
                "declared on the <b>Noise</b> tab.  Use whichever is more "
                "natural &mdash; the noise tab is mandatory only for "
                "colored / cross-correlated noise."
                '</p>'),
            self._w_action,
        ])

        # Tab 8: Mean-field equations — DAE form.
        # One row per equation (residual ``LHS - RHS = 0``).  Dt allowed
        # on the LHS; absent ⇒ algebraic.  At MF the DAE solver sets
        # Dt → 0; for stability it keeps Dt symbolic.  Population
        # determines the index range for ``i`` on both sides.
        self._tbl_mfeqs = DynamicTable(
            columns=[
                {'name': 'lhs',        'kind': 'text',
                 'placeholder': '(Dt + mu) * x',
                 'width': '240px'},
                {'name': 'rhs',        'kind': 'text',
                 'placeholder': '-eps * x^3   (or 0 for a linear EOM)',
                 'width': '360px'},
                {'name': 'population', 'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '110px'},
            ],
            # No starter row.  The lhs/rhs cells already show their
            # example via placeholder text; pre-populating with an
            # actual equation forces users to delete it before typing
            # their own.  Click "+ add row" to add your first equation.
            initial=[],
        )
        # Default fixed-point index for the multi-root MF solver.
        # When the DAE system has multiple roots (a bistable theory,
        # say), this picks WHICH sorted root the diagrammatic expansion
        # uses.  Sorted ascending by the first declared physical
        # field's first population index.  ``0`` = lowest, ``1`` =
        # second-lowest, ...  Out-of-range values get clamped at run
        # time with a warning.  This default is written into METADATA;
        # the theory runner picks it up unless overridden per-run.
        self._w_fpi_default = W.BoundedIntText(
            value=0, min=0, max=99, step=1,
            description='Default fixed_point_index:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='280px'),
        )
        # Linear-stability toggle.  Off by default — theories that
        # integrate out their voltages have all-algebraic equations
        # (no Dt anywhere); the generalized-eigenvalue ``(σA + B)``
        # has A ≡ 0, every eigenvalue lands at infinity, and "linear
        # stability" has no physical meaning.  Bistable / differential
        # theories that want stability-based root filtering must check
        # this box explicitly.  Writes ``.stability_analysis(True)``
        # into the generated theory file when on.
        self._w_stability_on = W.Checkbox(
            value=False,
            description='Run linear stability analysis',
            indent=False,
            style={'description_width': 'initial'},
            layout=W.Layout(width='320px'),
        )
        # Optional explicit seed box for multi-start Newton, as a
        # Python dict literal mapping variable name → (low, high).
        # Empty → use domain-aware defaults (positive → [0, 5·scale],
        # real → [-3·scale, 3·scale]).  Used by solve_mean_field_dae's
        # ``seed_box`` kwarg.
        self._w_seed_box = W.Textarea(
            value='',
            placeholder=("Optional Python dict literal, e.g.\n"
                         "{'x': (-3.0, 3.0), 'y': (-3.0, 3.0)}"),
            layout=W.Layout(width='460px', height='60px'),
        )

        tab_mfeqs = W.VBox([
            W.HTML(
                '<h4>Mean-field equations</h4>'
                '<p style="color:#555;font-size:90%;">'
                "One row per equation.  The solver enforces "
                "<code>LHS = RHS</code> at the saddle.  Both sides are "
                "ordinary expressions in your declared fields and "
                "parameters; <code>Dt</code> stands for the time-"
                "derivative operator (it gets set to 0 when computing "
                "the saddle, and kept symbolic when checking stability).  "
                "<br><br>"
                "Examples:"
                "<ul style='margin-top:2px;'>"
                "<li><code>(Dt + mu) * x</code> = <code>0</code> "
                "&nbsp;&rarr;&nbsp; the saddle of <code>dx/dt = -mu*x</code>"
                "</li>"
                "<li><code>(Dt + mu) * x</code> = <code>-eps * x^3</code> "
                "&nbsp;&rarr;&nbsp; cubic well; the solver finds all roots</li>"
                "</ul>"
                "If a field is indexed by a population, use "
                "<code>x[i]</code> on both sides and select that "
                f"population in the dropdown.  For a scalar field, "
                f"leave the population as <code>{_NONE}</code>."
                '</p>'),
            self._tbl_mfeqs.show(),
            W.HTML(
                '<br><h4>Picking a root when there are several</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Some equations have more than one solution (e.g. a "
                "double-well potential).  The solver finds them all and "
                "sorts them in ascending order; "
                "<code>fixed_point_index</code> picks which one to "
                "expand around (0 = smallest, 1 = next, &hellip;)."
                "<br><br>"
                "<b>Stability filter</b> (the checkbox below).  "
                "<u>ON</u>: keep only the linearly stable roots before "
                "indexing.  This is what you want for a bistable theory "
                "where you only care about the stable branch.  "
                "<u>OFF</u> (default): use every root.  Leave OFF when "
                "none of your equations contain <code>Dt</code> "
                "(purely algebraic / stationary system) &mdash; there's "
                "nothing for the stability test to score, so the filter "
                "would be meaningless."
                '</p>'),
            self._w_fpi_default,
            self._w_stability_on,
            W.HTML(
                '<br><h4>Where to look for roots (optional)</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Newton's method needs starting guesses.  By default the "
                "framework samples sensible ranges based on each "
                "variable's <code>domain</code>: <code>positive</code> "
                "&rarr; <code>[0, 5&middot;scale]</code>, "
                "real&nbsp;&rarr;&nbsp;<code>[&minus;3&middot;scale, 3&middot;scale]</code> "
                "<code>real</code> &rarr; "
                "<code>[&minus;3&middot;scale, 3&middot;scale]</code>.  "
                "Only set this if you know the default range misses a "
                "root you care about (e.g. a bounded state living in "
                "<code>(&minus;1, 1)</code> when the auto-range is too "
                "wide)."
                '</p>'),
            self._w_seed_box,
        ])

        # Tab 9: Defaults (run metadata only — default-fundamental
        # values are now sourced from each parameter's ``default=...``
        # declaration on the Parameters tab, so a separate "default
        # fundamental" textarea would be redundant and a foot-gun for
        # drift).
        #
        # Structured-widget redesign (2026-05-26): the previous free-
        # form Python-dict textarea was a foot-gun — a single typo
        # (missing comma, smart-quote substitution from copy-paste,
        # etc.) routed the user through ``ast.literal_eval`` to a
        # cryptic ``SyntaxError``.  Each metadata key now has its own
        # typed widget; the dict is assembled in ``_collect_metadata``.
        self._w_k_default = W.BoundedIntText(
            value=2, min=1, max=5, step=1,
            description='k (cumulant order):',
            style={'description_width': '150px'},
            layout=W.Layout(width='250px'),
        )
        self._w_ell_default = W.BoundedIntText(
            value=0, min=0, max=3, step=1,
            description='max_ell (loop order):',
            style={'description_width': '150px'},
            layout=W.Layout(width='250px'),
        )
        self._w_tau_max = W.FloatText(
            value=50.0, step=1.0,
            description='tau_max:',
            style={'description_width': '150px'},
            layout=W.Layout(width='250px'),
        )
        self._w_tau_step = W.FloatText(
            value=0.5, step=0.1,
            description='tau_step:',
            style={'description_width': '150px'},
            layout=W.Layout(width='250px'),
        )
        # Recommended external fields — one row per leaf (field +
        # 1-based leaf index).  Up to k rows typically; UI lets users
        # add / remove freely.
        self._tbl_ext_fields = DynamicTable(
            columns=[
                {'name': 'field',       'kind': 'text',
                 'placeholder': 'x',    'width': '120px'},
                {'name': 'leaf_index',  'kind': 'int',
                 'default': 1,          'width': '100px'},
            ],
            initial=[
                # Two leaves on the same scalar field ``x`` matches the
                # default ``k_default = 2`` (auto-correlator) and is the
                # most common k-point setup the runner cares about.
                {'field': 'x', 'leaf_index': 1},
                {'field': 'x', 'leaf_index': 2},
            ],
        )
        tab_defaults = W.VBox([
            W.HTML(
                '<h4>Run defaults</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Suggestions the runner notebook picks up as starting "
                "values.  Each is overridable per-run.  Nothing here "
                "changes the physics &mdash; only what the runner "
                "computes by default."
                '</p>'
                '<p style="color:#555;font-size:90%;"><b>What each knob means:</b><ul>'
                '<li><b>k</b> &mdash; how many external legs the '
                'cumulant has.  <code>k=1</code> = mean (one-point), '
                '<code>k=2</code> = covariance / power spectrum (two-point), '
                '<code>k=3</code> = third cumulant, &hellip;</li>'
                '<li><b>max_ell</b> &mdash; loop order in the diagrammatic '
                'expansion.  <code>0</code> = tree-level only (LNA / '
                'linear response); <code>1</code> = +&nbsp;1-loop; '
                '<code>2</code> = +&nbsp;2-loop.  Higher = more accurate '
                'but exponentially more expensive.</li>'
                '<li><b>tau_max</b>, <b>tau_step</b> &mdash; the time-lag '
                'grid for two-point quantities.  The cumulant '
                '<code>C(&tau;)</code> is evaluated at '
                '<code>0, tau_step, 2&middot;tau_step, &hellip;, tau_max</code>.  '
                'Smaller step + larger max = denser / longer curve, '
                'proportionally slower.</li>'
                '</ul></p>'),
            self._w_k_default,
            self._w_ell_default,
            self._w_tau_max,
            self._w_tau_step,
            W.HTML(
                '<br><h4>External legs</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Which observables to correlate.  Add one row per "
                "external leg; the list length should equal <b>k</b> "
                "above.  Use the field's name (matching the Fields tab) "
                "and a 1-based <code>leaf_index</code> to label which "
                "leg you mean.  Two legs on the same field "
                "(<code>x[1], x[2]</code>) compute the <i>auto</i>-"
                "correlation; legs on different fields "
                "(<code>x[1], y[1]</code>) compute a <i>cross</i>-"
                "correlation."
                '</p>'),
            self._tbl_ext_fields.show(),
        ])

        # Compose into Tab widget
        self._tabs = W.Tab(children=[
            tab_model, tab_populations, tab_fields, tab_params,
            tab_functions, tab_kernels, tab_cgfs, tab_action,
            tab_mfeqs, tab_defaults,
        ])
        for i, title in enumerate([
            '1. Model', '2. Populations', '3. Fields', '4. Parameters',
            '5. Functions', '6. Kernels', '7. Noise', '8. Action',
            '9. MF', '10. Defaults',
        ]):
            self._tabs.set_title(i, title)

        # ── Live wiring: when the Populations tab changes, refresh
        # every other tab's population-aware dropdowns and re-template
        # the parameter / kernel default cells.
        def _on_populations_changed():
            self._tbl_physical.refresh_dropdown_options('population')
            self._tbl_parameters.refresh_dropdown_options('index_1')
            self._tbl_parameters.refresh_dropdown_options('index_2')
            self._tbl_kernels.refresh_dropdown_options('index_1')
            self._tbl_kernels.refresh_dropdown_options('index_2')
            self._tbl_mfeqs.refresh_dropdown_options('population')
            self._autofill_default_templates()
        self._tbl_populations.on_change(_on_populations_changed)

        # Whenever a parameter's index_1 / index_2 changes, retemplate
        # its default cell.
        self._tbl_parameters.on_change(self._autofill_default_templates)

        # ── Bottom buttons + status ───────────────────────────────
        self._w_save_path = W.Text(
            value='', placeholder='leave blank → auto-name from theory name',
            layout=W.Layout(width='420px'),
            description='Save to:',
            style={'description_width': '80px'},
        )
        self._btn_preview = W.Button(description='Preview .theory.py',
                                     button_style='', icon='eye',
                                     layout=W.Layout(width='180px'))
        self._btn_save    = W.Button(description='Save theory file',
                                     button_style='success', icon='save',
                                     layout=W.Layout(width='180px'))
        # Pre-compute essentials: one-shot structural validation +
        # propagator + saddle cache.  Lives next to Save because it's
        # the "is this theory healthy" check you run right after
        # saving (the file isn't strictly required — precompute reads
        # the in-memory spec).  Status output goes to the same global
        # ``self._status`` panel the save / load buttons use.
        self._btn_precompute = W.Button(description='Pre-compute',
                                        button_style='info',
                                        icon='check-circle',
                                        tooltip=(
                                            'Expand at taylor_order=2, '
                                            'verify mean-field saddle, '
                                            'build + cache propagator.'),
                                        layout=W.Layout(width='160px'))
        self._btn_reset   = W.Button(description='Reset',
                                     button_style='warning',
                                     layout=W.Layout(width='100px'))
        self._btn_preview.on_click(self._on_preview)
        self._btn_save.on_click(self._on_save)
        self._btn_precompute.on_click(self._on_precompute)
        self._btn_reset.on_click(self._on_reset)

        # Load existing theory: dropdown of files in theories_dir,
        # plus a Load button.  The dropdown auto-refreshes on each
        # _on_load click so a freshly-saved theory shows up without
        # restarting the UI.
        self._w_load_pick = W.Dropdown(
            options=self._list_theory_files(),
            value=None,
            description='Load:',
            style={'description_width': '60px'},
            layout=W.Layout(width='420px'),
        )
        self._btn_load = W.Button(description='Load theory file',
                                  button_style='info', icon='upload',
                                  layout=W.Layout(width='180px'))
        self._btn_load.on_click(self._on_load)

        self._status = W.Output(
            layout=W.Layout(border='1px solid #ddd', padding='6px',
                            min_height='80px', max_height='420px',
                            overflow='auto'))

        # Header
        self._header = W.HTML(
            '<h2 style="margin-bottom:4px;">MSR-JD Theory Builder</h2>'
            '<p style="margin-top:0;color:#555;">'
            'Fill in the tabs below.  Hit <b>Save theory file</b> to write a '
            '<code>.theory.py</code> in <code>theories/</code> — '
            '<code>notebooks/theory_runner.ipynb</code> picks it up '
            'automatically.  Use <b>Load theory file</b> to re-open a '
            'saved theory for editing.'
            '</p>')
        self._header.add_class('tb-header')

        # Validation sidebar — re-rendered on every cross-tab change.
        # Lives between the header and the tab strip so it's always
        # visible no matter which tab the user is on.  See
        # ``_refresh_validation``.
        self._validation_panel = W.HTML(value='')
        self._validation_panel.add_class('tb-validation')

        # Apply CSS classes to widgets we want themed.
        self._tabs.add_class('tb-tabs')
        # Each tab body gets the card panel styling.
        for child in self._tabs.children:
            try:
                child.add_class('tb-tab-panel')
            except Exception:
                pass
        # Action textarea uses the code-font palette.
        try:
            self._w_action._text_w.add_class('tb-code-area')
        except Exception:
            pass
        # Seed-box textarea also code-font.
        try:
            self._w_seed_box.add_class('tb-code-area')
        except Exception:
            pass
        # Dynamic-table header rows and bodies get striping.
        for tbl in (self._tbl_populations, self._tbl_physical,
                    self._tbl_parameters, self._tbl_functions,
                    self._tbl_kernels, self._tbl_cgfs, self._tbl_mfeqs,
                    self._tbl_ext_fields):
            try:
                tbl._header.add_class('tb-table-header')
                tbl._rows_container.add_class('tb-table-body')
            except Exception:
                pass
        # Status panel.
        self._status.add_class('tb-status')

        # Button-palette classes (replaces the inconsistent
        # ``button_style=`` ad-hoc settings).  Each class maps to one of
        # the four-button palette defined in ``_THEORY_BUILDER_CSS``.
        self._btn_save.add_class('tb-btn-primary')
        self._btn_save.button_style = ''     # let CSS win
        self._btn_preview.add_class('tb-btn-secondary')
        self._btn_preview.button_style = ''
        self._btn_precompute.add_class('tb-btn-secondary')
        self._btn_precompute.button_style = ''
        self._btn_reset.add_class('tb-btn-danger')
        self._btn_reset.button_style = ''
        self._btn_load.add_class('tb-btn-link')
        self._btn_load.button_style = ''

        self._root = W.VBox([
            self._header,
            self._validation_panel,
            self._tabs,
            self._build_bottom_bar(),
            self._status,
        ])
        self._root.add_class('tb-root')

        # ── Cross-cutting wiring ──────────────────────────────────
        # Every table (and every standalone widget below) calls
        # ``_mark_changed`` on edit, which:
        #   1. flips ``self._dirty = True`` (gates Reset / Load behind
        #      the confirm-discard checkbox);
        #   2. re-renders the validation + cheat-sheet sidebar.
        for tbl in (self._tbl_populations, self._tbl_physical,
                    self._tbl_parameters, self._tbl_functions,
                    self._tbl_kernels, self._tbl_cgfs, self._tbl_mfeqs,
                    self._tbl_ext_fields):
            try:
                tbl.on_change(self._mark_changed)
            except Exception:
                pass
        for w in (self._w_name, self._w_description,
                  self._w_fpi_default, self._w_stability_on,
                  self._w_seed_box,
                  self._w_k_default, self._w_ell_default,
                  self._w_tau_max, self._w_tau_step):
            try:
                w.observe(lambda _c: self._mark_changed(), names='value')
            except Exception:
                pass
        try:
            self._w_action._text_w.observe(
                lambda _c: self._mark_changed(), names='value')
        except Exception:
            pass
        # Save-path field: update the derived-filename hint as the user
        # types the theory name.
        try:
            self._w_name.observe(
                lambda _c: self._refresh_save_hint(), names='value')
            self._w_save_path.observe(
                lambda _c: self._refresh_save_hint(), names='value')
        except Exception:
            pass
        # Initial render of derived-filename hint + validation panel.
        self._refresh_save_hint()

    # ── Bottom action bar ─────────────────────────────────────────
    def _build_bottom_bar(self) -> W.VBox:
        """Compose the bottom button rail: save-path field with derived-
        filename hint, primary actions, the unsaved-changes gate, and
        the Load row.  Returns the VBox; widgets are stored on self.

        Replaces the older two-HBox layout (see prior version of
        ``_build_widgets`` — the bottom 6 lines of it).  Lays out:

          [ save_path | hint ]   ← derived filename preview
          [ Save | Preview | Pre-compute | Open in runner ]
          [ Reset+confirm | Load+pick+confirm ]
        """
        # Save-path hint label (updated by ``_refresh_save_hint``).
        self._save_hint = W.HTML(value='')
        self._save_hint.add_class('tb-save-hint')

        # "Open in runner" bridge (item 5).  After Save, the user can
        # click this to open ``theory_runner.ipynb`` and have the just-
        # saved THEORY_NAME picked up automatically via a tiny scratch
        # file written by ``_on_save``.
        self._btn_open_runner = W.Button(
            description='Open in runner notebook',
            icon='external-link',
            layout=W.Layout(width='220px'),
            tooltip='Open theory_runner.ipynb with the last-saved '
                    'theory pre-selected.')
        self._btn_open_runner.add_class('tb-btn-link')
        self._btn_open_runner.button_style = ''
        self._btn_open_runner.on_click(self._on_open_in_runner)

        # Discard-changes confirm checkboxes for Reset / Load.
        # Gate the destructive action behind an explicit acknowledgement
        # whenever ``self._dirty == True``.  When the form is clean,
        # the checkbox is effectively a no-op.
        self._chk_confirm_reset = W.Checkbox(
            value=False, indent=False,
            description='discard unsaved changes',
            layout=W.Layout(width='220px'))
        self._chk_confirm_load = W.Checkbox(
            value=False, indent=False,
            description='discard unsaved changes',
            layout=W.Layout(width='220px'))

        row_path = W.HBox([self._w_save_path, self._save_hint])
        row_actions = W.HBox([
            self._btn_save, self._btn_preview,
            self._btn_precompute, self._btn_open_runner,
        ])
        row_reset = W.HBox([self._chk_confirm_reset, self._btn_reset])
        row_load = W.HBox([
            self._chk_confirm_load, self._w_load_pick, self._btn_load,
        ])
        return W.VBox([
            row_path,
            row_actions,
            W.HTML('<hr style="margin:12px 0;border:none;'
                   'border-top:1px solid #e5e7eb;">'),
            row_reset,
            row_load,
        ])

    # ── Validation + declared-names cheat sheet ───────────────────
    # ── Action-text validator ────────────────────────────────────────
    # Built-in identifiers that can appear in the action without being
    # declared by the user.  These are Sage / math primitives the
    # framework rebinds when parsing the action text.  Keeping the list
    # explicit (rather than allowlisting on demand) lets us catch typos
    # like ``hevyside`` or ``exp1`` rather than silently treating them
    # as unbound free variables.
    _ACTION_BUILTINS = frozenset({
        # Differential / convolution operators
        'Dt', 'Conv',
        # Standard math
        'sum', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
        'sqrt', 'abs', 'pi', 'e', 'I', 'oo',
        # Heaviside / dirac
        'heaviside', 'dirac_delta', 'sign',
        # Indexing helpers
        'range', 'len',
    })

    def _action_known_names(self, spec: dict) -> tuple[set[str], set[str]]:
        """Build (known_global_names, known_populations) for action validation.

        ``known_global_names``: every identifier that may legally appear
        as a free variable in the action expression — declared params,
        physical fields and their auto-derived ``<name>t`` / ``d<name>``
        / ``<name>star`` companions, kernels, functions, plus the
        :data:`_ACTION_BUILTINS` set.

        ``known_populations``: just the population names — used to
        validate the right-hand side of comprehensions
        (``for i in <pop>``).  The framework also rebinds each as
        ``pop_<name>``; both spellings are accepted.
        """
        names: set[str] = set(self._ACTION_BUILTINS)
        for p in spec.get('parameters') or []:
            nm = (p.get('name') or '').strip()
            if nm: names.add(nm)
        for f in spec.get('physical_fields') or []:
            nm = (f.get('name') or '').strip()
            if not nm: continue
            names.add(nm)           # physical field itself
            names.add(f'{nm}t')     # response field (auto)
            names.add(f'd{nm}')     # fluctuation (auto, sometimes used)
            names.add(f'{nm}star')  # saddle (rarely in action but allowed)
        for k in spec.get('kernels') or []:
            nm = (k.get('name') or '').strip()
            if nm: names.add(nm)
        for fn in spec.get('functions') or []:
            nm = (fn.get('name') or '').strip()
            if nm: names.add(nm)
        pop_set: set[str] = set()
        for pop in spec.get('populations') or []:
            nm = (pop.get('name') or '').strip()
            if nm:
                pop_set.add(nm)
                names.add(nm)          # legal in ``for i in <pop>``
                names.add(f'pop_{nm}')  # legal too
        names.add('pop')               # legacy alias for the only-pop case
        return names, pop_set

    def _validate_action(self, action_text: str,
                         spec: dict) -> list[tuple[str, str]]:
        """Parse the action and return ``[(severity, message), ...]``.

        Catches:
          * syntax errors (with the offending line number)
          * identifiers that aren't declared anywhere
          * ``for i in X`` where ``X`` isn't a declared population
          * indexed access ``x[i]`` where ``x`` is unknown or ``i`` is
            unbound and no comprehension provides it

        Comprehension-bound names are recursively tracked through
        nested ``for`` clauses so an inner index doesn't get reported
        as undeclared.
        """
        import ast
        text = (action_text or '').strip()
        if not text:
            return []   # empty action is its own finding upstream
        try:
            tree = ast.parse(text, mode='eval')
        except SyntaxError as e:
            line = (e.lineno or 1)
            col = (e.offset or 0)
            return [('error',
                     f'action syntax error at line {line}, col {col}: '
                     f'{e.msg}')]
        except Exception as e:
            return [('error', f'action parse failed: {type(e).__name__}: {e}')]

        known, pops = self._action_known_names(spec)

        # Walk the AST tracking comprehension scopes.  Each
        # comprehension introduces target names that are bound for the
        # duration of its iterator + body.
        problems: list[tuple[str, str]] = []
        unknown: set[str] = set()
        bad_pops: set[str] = set()

        def _collect_targets(node):
            """Yield names bound by an assignment/for-target."""
            if isinstance(node, ast.Name):
                yield node.id
            elif isinstance(node, (ast.Tuple, ast.List)):
                for e in node.elts:
                    yield from _collect_targets(e)

        def walk(node, bound: frozenset[str]):
            # Handle comprehension scopes specially: their generators
            # introduce names that the body and later generators see.
            if isinstance(node, (ast.GeneratorExp, ast.ListComp,
                                 ast.SetComp, ast.DictComp)):
                inner = set(bound)
                for gen in node.generators:
                    # Validate the iterator BEFORE the target binds —
                    # ``for i in foo`` needs ``foo`` to be known.
                    if isinstance(gen.iter, ast.Name):
                        nm = gen.iter.id
                        # Populations are the only legal iterators in
                        # the action's comprehension idiom.
                        if pops and nm not in pops and \
                                nm not in {f'pop_{p}' for p in pops} \
                                and nm != 'pop' and nm not in inner:
                            bad_pops.add(nm)
                        elif not pops and nm == 'pop':
                            # Special case: ``for i in pop`` with no
                            # populations declared — silent foot-gun.
                            problems.append(
                                ('error', f'action iterates over '
                                          f'"pop" but no populations '
                                          f'are declared'))
                    walk(gen.iter, frozenset(inner))
                    for tgt in _collect_targets(gen.target):
                        inner.add(tgt)
                    for cond in gen.ifs:
                        walk(cond, frozenset(inner))
                # Body sees all bound names.
                if isinstance(node, ast.DictComp):
                    walk(node.key, frozenset(inner))
                    walk(node.value, frozenset(inner))
                else:
                    walk(node.elt, frozenset(inner))
                return
            # Lambdas also introduce names.
            if isinstance(node, ast.Lambda):
                inner = set(bound) | {a.arg for a in node.args.args}
                walk(node.body, frozenset(inner))
                return
            # Plain Name: check against known + bound.
            if isinstance(node, ast.Name):
                if node.id not in known and node.id not in bound:
                    unknown.add(node.id)
            # Recurse into all child nodes.
            for child in ast.iter_child_nodes(node):
                walk(child, bound)

        walk(tree, frozenset())

        if unknown:
            shown = sorted(unknown)[:5]
            more = f' (+{len(unknown) - 5} more)' if len(unknown) > 5 else ''
            problems.append(
                ('error',
                 f'undeclared name(s) in action: '
                 + ', '.join(f'<code>{n}</code>' for n in shown)
                 + more))
        if bad_pops:
            shown = sorted(bad_pops)[:3]
            problems.append(
                ('error',
                 f'comprehension iterates over unknown population(s): '
                 + ', '.join(f'<code>{n}</code>' for n in shown)))
        return problems

    def _validate(self) -> dict:
        """Walk the current spec and emit per-tab status.

        Returns
        -------
        dict
            ``{tab_index_1based: [(severity, message), ...]}`` where
            severity ∈ {'warn', 'error'}.  Only emits findings that the
            user can ACT ON — not nags about defaults the user can
            clear with one period.  Re-issued on every cross-tab change.

        What we DO check
        ----------------
        * Field / parameter / kernel referencing an undeclared population
          (the silent ``n_populations=0`` bug class)
        * Action-text syntax + cross-reference against declared names
          (the actual high-value validation — catches typos and
          missing-declaration foot-guns at type time)
        * Defaults-tab dict literal parse failure (otherwise surfaces as
          an opaque traceback on Save)

        What we DON'T check
        -------------------
        * "Theory name still at default" — user can clear with one period;
          it's not an error, just a starting value.
        * "Stability analysis OFF" — that's a sensible default, not a
          mistake.  Stability OFF for an algebraic-MF (no ``Dt``) theory
          is correct.
        * "Action is blank" — the only reason to flag this is at Save
          time; emitting it on every keystroke is just noise.
        """
        out: dict[int, list[tuple[str, str]]] = {}
        def _add(tab, sev, msg):
            out.setdefault(tab, []).append((sev, msg))

        # Snapshot the form state.  Wrap in try because some _collect
        # paths can raise on malformed Defaults dict — in that case the
        # error itself becomes a validation finding.
        try:
            spec = self._collect()
            spec_error = None
        except Exception as exc:
            spec = {}
            spec_error = str(exc)

        if spec_error:
            _add(10, 'error', f'Defaults tab parse error: {spec_error}')
            return out

        # ── Tab 2 — Populations: silent ``n_populations=0`` bug class ──
        pops = spec.get('populations') or []
        pop_names = {p.get('name') for p in pops if p.get('name')}
        if not pops:
            referencing = []
            for f in spec.get('physical_fields') or []:
                if (f.get('population') or '').strip():
                    referencing.append(f"field '{f.get('name')}'")
            for p in spec.get('parameters') or []:
                if p.get('indexed_by'):
                    referencing.append(f"parameter '{p.get('name')}'")
            for k in spec.get('kernels') or []:
                if k.get('indexed_by'):
                    referencing.append(f"kernel '{k.get('name')}'")
            if referencing:
                _add(2, 'error',
                     'no populations declared but '
                     + ', '.join(referencing[:3])
                     + (' …' if len(referencing) > 3 else '')
                     + ' reference one')

        # ── Tab 3 — Fields: stale population references ──────────────
        fields = spec.get('physical_fields') or []
        for f in fields:
            pop = (f.get('population') or '').strip()
            if pop and pop_names and pop not in pop_names:
                _add(3, 'error',
                     f"field '{f.get('name')}' references unknown "
                     f"population '{pop}'")

        # ── Tab 4 — Parameters: stale population references ──────────
        for p in spec.get('parameters') or []:
            for tag in (p.get('indexed_by') or []):
                if pop_names and tag not in pop_names:
                    _add(4, 'error',
                         f"parameter '{p.get('name')}' indexed by "
                         f"unknown population '{tag}'")

        # ── Tab 6 — Kernels: stale population references ─────────────
        for k in spec.get('kernels') or []:
            for tag in (k.get('indexed_by') or []):
                if pop_names and tag not in pop_names:
                    _add(6, 'error',
                         f"kernel '{k.get('name')}' indexed by "
                         f"unknown population '{tag}'")

        # ── Tab 8 — Action: AST parse + cross-reference ──────────────
        for sev, msg in self._validate_action(
                spec.get('action_text') or '', spec):
            _add(8, sev, msg)

        return out

    def _build_cheat_sheet(self) -> str:
        """Return an HTML snippet listing all declared names by tab.

        Acts as a persistent reference for the user: as they declare
        populations / fields / parameters, those names appear here so
        the user doesn't have to bounce back to verify spelling when
        writing the action or MF equations.

        Shows the form (scalar vs indexed) appropriate to the CURRENT
        state:
          * No populations declared → fields shown as scalars
            (``n``, ``nt``, ``nstar`` — NO ``[i]``).
          * Populations declared → fields shown as indexed, with the
            explicit ``for i in pop_<X>`` so the user can copy-paste
            the comprehension idiom directly.

        Filters out rows where the user hasn't typed anything (an empty
        DynamicTable row is the default starting state; surfacing it as
        a "field named ''" entry is just noise).
        """
        try:
            spec = self._collect()
        except Exception:
            return ''

        # Populations: present?
        pops = [p for p in (spec.get('populations') or [])
                if (p.get('name') or '').strip()]
        pop_rows = [f"<code>{p['name']}</code> (size {p.get('size', 1)})"
                    for p in pops]

        # Fields: show scalar form when no pops exist for this field;
        # show indexed form + the comprehension hint when they do.
        field_rows = []
        for f in (spec.get('physical_fields') or []):
            name = (f.get('name') or '').strip()
            if not name:
                continue   # skip blank starter rows
            field_pop = (f.get('population') or '').strip()
            # An indexed field needs both: a non-blank population AND
            # that population exists in the Populations tab.  Either
            # missing → render as scalar.
            valid_pop = (field_pop
                         and field_pop in {p['name'] for p in pops})
            if valid_pop:
                field_rows.append(
                    f"<code>{name}[i]</code> for "
                    f"<code>i in pop_{field_pop}</code>"
                    f" — response <code>{name}t[i]</code>,"
                    f" saddle <code>{name}star[i]</code>")
            else:
                # Scalar field — show without [i], note explicitly so
                # the user doesn't reach for the comprehension idiom.
                field_rows.append(
                    f"<code>{name}</code> <i>(scalar)</i>"
                    f" — response <code>{name}t</code>,"
                    f" saddle <code>{name}star</code>")

        param_rows = [f"<code>{p.get('name')}</code>"
                      for p in (spec.get('parameters') or [])
                      if (p.get('name') or '').strip()]
        kernel_rows = [f"<code>{k.get('name')}</code>"
                       for k in (spec.get('kernels') or [])
                       if (k.get('name') or '').strip()]
        function_rows = [
            f"<code>{fn.get('name')}({', '.join(fn.get('args') or [])})</code>"
            for fn in (spec.get('functions') or [])
            if (fn.get('name') or '').strip()
        ]

        sections = []
        if pop_rows:
            sections.append('<b>Populations:</b> ' + ', '.join(pop_rows))
        if field_rows:
            sections.append('<b>Fields:</b><ul>'
                            + ''.join(f'<li>{r}</li>' for r in field_rows)
                            + '</ul>')
        if param_rows:
            sections.append('<b>Parameters:</b> ' + ', '.join(param_rows))
        if kernel_rows:
            sections.append('<b>Kernels:</b> ' + ', '.join(kernel_rows))
        if function_rows:
            sections.append('<b>Functions:</b> ' + ', '.join(function_rows))
        if not sections:
            return '<i>Nothing declared yet.</i>'
        return '<br>'.join(sections)

    def _refresh_validation(self) -> None:
        """Re-render the validation + cheat-sheet sidebar.

        Wired to every cross-tab change via the ``_mark_changed``
        callback.  Cheap (a single ``_collect()`` + string-build) — no
        async / debounce needed.
        """
        tab_titles = [
            '1. Model', '2. Populations', '3. Fields', '4. Parameters',
            '5. Functions', '6. Kernels', '7. Noise', '8. Action',
            '9. MF', '10. Defaults',
        ]
        findings = self._validate()
        # Top: per-tab readiness badges
        badges = []
        for i, t in enumerate(tab_titles, start=1):
            f = findings.get(i, [])
            if any(sev == 'error' for sev, _ in f):
                color = '#b91c1c'
                glyph = '●'
            elif any(sev == 'warn' for sev, _ in f):
                color = '#b45309'
                glyph = '●'
            else:
                color = '#15803d'
                glyph = '●'
            badges.append(
                f'<span style="color:{color};margin-right:8px;" '
                f'title="{t}">{glyph} {t}</span>')
        badge_strip = ' '.join(badges)

        # Below: detailed list of warnings/errors.  Only show the
        # "looks ready" line when SOMETHING is declared — chirping a
        # green tick over a blank starter form is noise.
        items = []
        for tab_idx, msgs in sorted(findings.items()):
            for sev, msg in msgs:
                cls = {'ok': 'tb-v-ok',
                       'warn': 'tb-v-warn',
                       'error': 'tb-v-error'}.get(sev, '')
                items.append(
                    f'<li><span class="{cls}">'
                    f'[{tab_titles[tab_idx-1]}]</span> {msg}</li>')
        if items:
            detail = '<ul>' + ''.join(items) + '</ul>'
        else:
            # Only confirm "ready" once the user has typed at least one
            # field / parameter / action — otherwise the blank-form
            # default makes the sidebar lie.
            try:
                _spec = self._collect()
                has_content = (
                    bool(_spec.get('physical_fields')) or
                    bool(_spec.get('parameters')) or
                    (_spec.get('action_text') or '').strip()
                )
            except Exception:
                has_content = False
            detail = ('<span class="tb-v-ok">'
                      'No problems detected.</span>') if has_content else ''

        cheat = self._build_cheat_sheet()
        cheat_html = (f'<div style="margin-top:8px;padding-top:8px;'
                      f'border-top:1px solid #e5e7eb;">'
                      f'<b>Declared so far:</b><br>{cheat}</div>'
                      if cheat else '')

        html = (
            f'<b>Readiness:</b> {badge_strip}<br>'
            f'{detail}'
            f'{cheat_html}'
        )
        self._validation_panel.value = html

    def _refresh_save_hint(self) -> None:
        """Update the derived-filename hint next to the save-path field.

        Shows the path the file would land at if the user clicked Save
        right now — so they see the slugification BEFORE writing.
        """
        if not hasattr(self, '_save_hint'):
            return
        # Mirror the logic in ``_on_save``.
        explicit = (self._w_save_path.value or '').strip()
        if explicit:
            if os.path.isabs(explicit) or explicit.endswith('.py'):
                path = explicit
            else:
                path = os.path.join(self.theories_dir, explicit)
        else:
            # Auto-slugify from the theory name.
            from pipeline.theory_serialize import _slugify
            try:
                slug = _slugify(self._w_name.value or '')
            except Exception:
                slug = 'theory'
            path = os.path.join(self.theories_dir, f'{slug}.theory.py')
        rel = os.path.relpath(path, self.theories_dir)
        # Tidy display: ``theories/foo.theory.py`` rather than absolute.
        self._save_hint.value = f'→ writes <code>theories/{rel}</code>'

    # ── Dirty / unsaved-changes tracking ──────────────────────────
    def _mark_changed(self) -> None:
        """Mark the form as dirty AND re-render the validation panel.

        Single-line callback fired by every table on_change and every
        standalone widget observe.  Keeps the two side-effects
        (dirty bookkeeping + sidebar refresh) consistent.
        """
        self._dirty = True
        try:
            self._refresh_validation()
        except Exception:
            pass

    def _check_dirty_guard(self, which: str) -> bool:
        """If the form has unsaved changes, the matching
        confirm checkbox must be checked.  Returns True if the action
        is allowed to proceed.
        """
        if not getattr(self, '_dirty', False):
            return True
        chk = (self._chk_confirm_reset if which == 'reset'
               else self._chk_confirm_load)
        if chk and chk.value:
            return True
        return False

    # ── Open in runner notebook (item 5) ──────────────────────────
    def _on_open_in_runner(self, _btn) -> None:
        """Write a tiny scratch file with the last-saved theory name
        and emit JS to open ``theory_runner.ipynb`` in a new tab.

        The runner notebook reads ``.theories/.last_built`` if
        ``THEORY_NAME`` is left blank.  Gracefully degrades to a
        printed instruction when JS escape isn't available (e.g.
        nbconvert export).
        """
        from IPython.display import Javascript
        self._status.clear_output()
        with self._status:
            if not self.last_saved:
                print('[runner] no theory has been saved yet — '
                      'click Save first.')
                return
            slug = os.path.basename(
                self.last_saved)[:-len('.theory.py')]
            scratch_dir = os.path.join(self.theories_dir, '..',
                                       '.theories')
            os.makedirs(scratch_dir, exist_ok=True)
            scratch_path = os.path.join(scratch_dir, '.last_built')
            with open(scratch_path, 'w') as fh:
                fh.write(slug + '\n')
            print(f'[runner] wrote scratch file → {scratch_path}')
            print(f'[runner] THEORY_NAME = {slug!r}')
            try:
                display(Javascript(
                    "window.open('theory_runner.ipynb', '_blank');"))
                print('[runner] opening theory_runner.ipynb in a new '
                      'tab…')
            except Exception:
                print('[runner] (could not open via JS; navigate to '
                      'notebooks/theory_runner.ipynb manually)')

    # ── Humanized error messages ──────────────────────────────────
    def _humanize_error(self, exc: BaseException, context: str) -> str:
        """Turn an exception into a single human-readable line.

        Covers the most common errors the user hits when their action
        / MF equation / Defaults dict has a problem.  Falls back to
        ``repr(exc)`` for the long tail.

        Parameters
        ----------
        exc : BaseException
            The raised exception.
        context : str
            One of 'save', 'preview', 'precompute' — used to tailor
            the prefix.
        """
        cls = type(exc).__name__
        msg = str(exc)
        prefix = {'save':       '[save] ',
                  'preview':    '[preview] ',
                  'precompute': '[precompute] ',
                  }.get(context, '')
        # SyntaxError carries lineno / offset on a multiline source —
        # likely the action textarea.
        if isinstance(exc, SyntaxError):
            line = getattr(exc, 'lineno', None)
            col = getattr(exc, 'offset', None)
            where = (f' at line {line}' if line else '') + \
                    (f', column {col}' if col else '')
            return (f'{prefix}Syntax error in action / equation '
                    f'expression{where}: {msg}')
        # NameError = user typed a parameter or field name that
        # wasn't declared.
        if isinstance(exc, NameError):
            m = msg
            return (f'{prefix}Undefined name — {m}.  Declare it on '
                    f'the Parameters / Fields / Kernels tab first.')
        if isinstance(exc, KeyError):
            return (f'{prefix}Missing required field: {msg}')
        if isinstance(exc, ValueError):
            return f'{prefix}{msg}'
        # Generic fallback — short class+message.
        return f'{prefix}{cls}: {msg}'

    # ── Public API ────────────────────────────────────────────────
    def show(self) -> None:
        """Render the form in the current notebook cell.

        Injects the inline CSS overhaul on first call per session, then
        displays the root VBox.  Re-rendering the same instance is
        cheap — the CSS is module-guarded so it only ships once.
        """
        global _CSS_INJECTED
        if not _CSS_INJECTED:
            display(HTML(_THEORY_BUILDER_CSS))
            _CSS_INJECTED = True
        # Make sure the spec is validated once on first render so the
        # sidebar shows real state rather than placeholder text.
        try:
            self._refresh_validation()
        except Exception:
            pass
        display(self._root)

    def spec(self) -> dict:
        """Snapshot the form as a serializer-ready spec dict."""
        return self._collect()

    # ── Form-state collection ─────────────────────────────────────
    def _collect(self) -> dict:
        # Parse default-fundamental and metadata as Python literals.
        def _eval_dict(text: str, label: str) -> dict:
            text = (text or '').strip()
            if not text:
                return {}
            try:
                obj = eval(text, {'__builtins__': {}}, {})
            except Exception as e:
                raise ValueError(f'Could not parse {label}: {e}')
            if not isinstance(obj, dict):
                raise ValueError(f'{label} must be a dict literal; got '
                                 f'{type(obj).__name__}')
            return obj

        _NONE = '—'

        def _index_list(row: dict) -> list[str]:
            """Translate (index_1, index_2) dropdowns into the
            ``indexed_by`` list the serializer wants: empty list for
            scalar, [pop] for vector, [pop1, pop2] for matrix."""
            i1 = (row.get('index_1') or '').strip()
            i2 = (row.get('index_2') or '').strip()
            out = []
            if i1 and i1 != _NONE:
                out.append(i1)
            if i2 and i2 != _NONE:
                out.append(i2)
            return out

        # Populations — name + size, plus optional description.
        populations = []
        for r in self._tbl_populations.get_rows():
            name = (r.get('name') or '').strip()
            if not name:
                continue
            try:
                size = int(r.get('size') or 0)
            except (TypeError, ValueError):
                size = 0
            entry = {'name': name, 'size': max(size, 1)}
            desc = (r.get('description') or '').strip()
            if desc:
                entry['description'] = desc
            populations.append(entry)

        # Physical fields — name + population.
        physical_fields = []
        for r in self._tbl_physical.get_rows():
            name = (r.get('name') or '').strip()
            if not name:
                continue
            entry = {'name': name}
            pop = (r.get('population') or '').strip()
            if pop:
                entry['population'] = pop
            # ``latex`` is no longer a UI input (column dropped); we
            # preserve any latex string already on a loaded spec by
            # carrying it through ``_loaded_extras`` (see ``load``).
            saved_latex = self._loaded_extras.get('field_latex', {}).get(name)
            if saved_latex:
                entry['latex'] = saved_latex
            desc = (r.get('description') or '').strip()
            if desc:
                entry['description'] = desc
            physical_fields.append(entry)

        # Parameters — name + indexed_by list (from dropdowns) +
        # optional domain + default.
        params = []
        for row in self._tbl_parameters.get_rows():
            name = (row.get('name') or '').strip()
            if not name:
                continue
            entry = {'name': name}
            idx = _index_list(row)
            if idx:
                entry['indexed_by'] = idx
            dom = (row.get('domain') or '').strip()
            if dom:
                entry['domain'] = dom
            d = (row.get('default') or '').strip()
            if d:
                try:
                    entry['default'] = eval(d, {'__builtins__': {}}, {})
                except Exception:
                    entry['default'] = d
            desc = (row.get('description') or '').strip()
            if desc:
                entry['description'] = desc
            params.append(entry)

        # Functions: split args by comma; pick up population annotation.
        functions = []
        for row in self._tbl_functions.get_rows():
            args = [a.strip() for a in (row.get('args') or '').split(',')
                    if a.strip()]
            entry = {'name': row['name'], 'args': args,
                     'expression': row.get('expression', '')}
            pop = (row.get('population') or '').strip()
            if pop and pop != _NONE:
                entry['population'] = pop
            # ``latex`` no longer a UI input — round-trip preserved via
            # _loaded_extras (see ``load``).
            saved_latex = self._loaded_extras.get(
                'function_latex', {}).get(entry['name'])
            if saved_latex:
                entry['latex'] = saved_latex
            desc = row.get('description')
            if desc:
                entry['description'] = desc
            functions.append(entry)

        # Kernels — name + indexed_by + time_expr/freq_image.
        kernels = []
        for row in self._tbl_kernels.get_rows():
            name = (row.get('name') or '').strip()
            if not name:
                continue
            entry = {'name': name}
            idx = _index_list(row)
            if idx:
                entry['indexed_by'] = idx
            for k in ('time_expr', 'freq_image'):
                v = (row.get(k) or '').strip()
                if v:
                    entry[k] = v
            # ``latex_name`` no longer a UI input — round-trip preserved
            # via _loaded_extras (see ``load``).
            saved_latex = self._loaded_extras.get(
                'kernel_latex', {}).get(name)
            if saved_latex:
                entry['latex_name'] = saved_latex
            kernels.append(entry)

        return {
            'name':            self._w_name.value,
            'populations':     populations,
            'n_populations':   len(populations),    # legacy, derived
            'description':     self._w_description.value,
            # Response fields are auto-generated by the framework from
            # the physical fields — no UI tab for them.
            'response_fields': [],
            'physical_fields': physical_fields,
            'parameters':      params,
            'functions':       functions,
            'kernels':         kernels,
            # CGF rows.  The ``response_legs`` cell holds either a
            # single response-field name (legacy single-leg cumulant)
            # or a comma-separated list (cross-field cumulant — leg k
            # sits on field name k of the list).  We normalise to the
            # spec keys downstream code expects: ``response_legs``
            # (list, when comma-separated) AND ``response_field``
            # (single string, for back-compat with old loaders).
            'cgf_terms':       _normalize_cgf_rows(self._tbl_cgfs.get_rows()),
            'action_text':     self._w_action.get_value(),
            # New DAE form: list of {lhs, rhs, population} records,
            # consumed by ``render_theory_file`` to emit
            # ``.equation(...)`` calls.  Legacy specs that came in via
            # ``set_mf_equation`` get re-rendered as ``.equation(...)``
            # calls too, with population back-inferred from the saddle
            # name (see ``load_spec_from_file``).
            'equations':       [
                {'lhs':        r['lhs'],
                 'rhs':        r['rhs'],
                 'population': (None if r['population'] in (_NONE, '', None)
                                else r['population'])}
                for r in self._tbl_mfeqs.get_rows()
                if (r.get('lhs') or '').strip() and (r.get('rhs') or '').strip()
            ],
            # Stability-analysis toggle on the MF tab.  Default OFF —
            # set when the user checks the box; theories that integrate
            # out voltages (all-algebraic equations) should leave it
            # off, bistable / differential theories that want
            # stability-based root selection should turn it on.
            'stability_analysis':  bool(self._w_stability_on.value),
            # default_fundamental is no longer a separate UI input —
            # the per-parameter ``default=...`` declarations on the
            # Parameters tab carry the suggested numerical values, and
            # the runner reads them from there.  Emit an empty dict
            # so the spec shape is preserved for legacy loaders.
            'default_fundamental': {},
            'metadata':            self._collect_metadata(),
        }

    def _collect_metadata(self) -> dict:
        """Build the METADATA dict from the per-key structured widgets
        on the Defaults tab + the dedicated MF-tab widgets.

        Each field has its own typed widget — the older free-form
        Python-dict textarea was deleted because ``ast.literal_eval``
        on hand-typed input produced cryptic syntax errors on smart-
        quote substitution / missing commas, with no recovery path.
        """
        import ast

        def _safe_dict(text, label):
            text = (text or '').strip()
            if not text:
                return {}
            try:
                val = ast.literal_eval(text)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f'{label} field is not a valid Python literal: {e}'
                ) from e
            if not isinstance(val, dict):
                raise ValueError(
                    f'{label} field must be a dict literal, got '
                    f'{type(val).__name__}'
                )
            return val

        md: dict = {
            'k_default':   int(self._w_k_default.value),
            'ell_default': int(self._w_ell_default.value),
            'tau_max':     float(self._w_tau_max.value),
            'tau_step':    float(self._w_tau_step.value),
        }
        # Recommended external fields → list of (name, idx) tuples.
        ext_rows = []
        for r in self._tbl_ext_fields.get_rows():
            name = (r.get('field') or '').strip()
            if not name:
                continue
            try:
                idx = int(r.get('leaf_index') or 1)
            except (TypeError, ValueError):
                idx = 1
            ext_rows.append((name, idx))
        if ext_rows:
            md['recommended_external_fields'] = ext_rows

        # Always write the default fixed-point index — even when it's 0
        # — so reload picks it up unambiguously.
        md['fixed_point_index_default'] = int(self._w_fpi_default.value)
        seed_text = (self._w_seed_box.value or '').strip()
        if seed_text:
            try:
                seed_box = _safe_dict(seed_text, 'seed_box_default')
            except Exception:
                # Bad input — fall through to writing the raw string;
                # preview will surface the problem.
                md['seed_box_default'] = seed_text
            else:
                if seed_box:
                    md['seed_box_default'] = {
                        k: tuple(v) if isinstance(v, (list, tuple)) else v
                        for k, v in seed_box.items()
                    }
        else:
            md.pop('seed_box_default', None)
        return md

    # ── Button handlers ───────────────────────────────────────────
    def _on_preview(self, _btn) -> None:
        self._status.clear_output()
        with self._status:
            try:
                spec = self._collect()
                src = render_theory_file(spec)
                print(src)
            except Exception as e:
                print(self._humanize_error(e, 'preview'))

    def _on_save(self, _btn) -> None:
        self._status.clear_output()
        with self._status:
            try:
                spec = self._collect()
            except Exception as e:
                print(self._humanize_error(e, 'save'))
                return
            try:
                target = (self._w_save_path.value.strip()
                          or self.theories_dir)
                if not os.path.isabs(target) and not target.endswith('.py'):
                    target = os.path.join(self.theories_dir, target)
                path = save_theory_to_file(spec, target)
            except Exception as e:
                print(self._humanize_error(e, 'save'))
                return
            self.last_saved = path
            self._dirty = False
            # Reset the dirty-guard checkboxes so a subsequent Load/
            # Reset doesn't surprise the user with a stale-acknowledged
            # confirm.
            try:
                self._chk_confirm_reset.value = False
                self._chk_confirm_load.value = False
            except Exception:
                pass
            print(f'[OK] Wrote {path}')
            rel = os.path.basename(path)[:-len('.theory.py')]
            print(f'\nIn notebooks/theory_runner.ipynb set:')
            print(f'    THEORY_NAME = {rel!r}')
            print(f"\nOr click 'Open in runner notebook' to launch the "
                  f"runner with this theory pre-selected.")
            # Refresh validation so the user sees the new save status.
            try:
                self._refresh_validation()
            except Exception:
                pass

    def _on_reset(self, _btn) -> None:
        # Gate destructive reset behind the confirm-discard checkbox
        # when there are unsaved changes.
        if not self._check_dirty_guard('reset'):
            self._status.clear_output()
            with self._status:
                print('[reset] You have unsaved changes.  Tick the '
                      "'discard unsaved changes' checkbox next to "
                      'Reset to confirm, or click Save first.')
            return
        # Re-build the whole UI to wipe state cleanly.
        self._build_widgets()
        self._dirty = False
        self.show()

    def _on_precompute(self, _btn) -> None:
        """Run ``pipeline.precompute(model)`` on the current UI state.

        Builds an in-memory ``model`` from the current spec (without
        writing a ``.theory.py`` file), then invokes the pre-compute
        primitive.  Output goes to the global ``self._status`` panel,
        same as Save / Load.
        """
        from pipeline import precompute
        from pipeline.theory_serialize import render_theory_file

        self._status.clear_output()
        with self._status:
            try:
                spec = self._collect()
            except Exception as e:
                print(self._humanize_error(e, 'precompute'))
                return

            # In-memory build: render to source, exec, call build().
            # This sidesteps writing a temp .theory.py just for a
            # validation pass.  ``render_theory_file`` already produces
            # a syntactically-valid module-level source.
            src = render_theory_file(spec)
            ns: dict = {}
            try:
                exec(compile(src, '<precompute-spec>', 'exec'), ns)
                model = ns['build']()
            except Exception as e:
                print(self._humanize_error(e, 'precompute'))
                print('  (model build failed — fix the offending row '
                      'above and retry)')
                return

            print(f'[precompute] model: {model.get("name", "<unnamed>")!r}')
            try:
                result = precompute(model, verbose=True)
            except Exception as e:
                print(self._humanize_error(e, 'precompute'))
                return

            # Concise summary at the bottom.
            print()
            print('─' * 60)
            ok = result.get('mf_check') == 'PASS'
            tag = '✓ PASS' if ok else f'✗ {result.get("mf_check")}'
            print(f'  MF check:       {tag}')
            print(f'  Saddle:         {dict(result.get("mf_values") or {})}')
            print(f'  Propagator:     '
                  f'{"cached" if result.get("propagator_built") else "FAILED"}')
            print(f'  Cache dir:      {result.get("cache_dir")}')
            print(f'  Wall time:      {result.get("wall_seconds", 0):.2f}s')

    # ── Loading existing theories ─────────────────────────────────
    def _list_theory_files(self) -> list[str]:
        """Return the list of ``*.theory.py`` filenames in
        ``self.theories_dir``, alphabetically sorted.  Used to
        populate the Load dropdown."""
        try:
            files = [f for f in os.listdir(self.theories_dir)
                     if f.endswith('.theory.py')]
        except OSError:
            files = []
        return sorted(files)

    def _on_load(self, _btn) -> None:
        """Read the file selected in the Load dropdown and repopulate
        the form widgets from its spec.  Gated behind the confirm-
        discard checkbox when there are unsaved changes."""
        if not self._check_dirty_guard('load'):
            self._status.clear_output()
            with self._status:
                print('[load] You have unsaved changes.  Tick the '
                      "'discard unsaved changes' checkbox next to "
                      'Load to confirm, or click Save first.')
            return
        # Refresh the file list so newly-saved theories appear without
        # restarting the UI.
        current = self._w_load_pick.value
        opts = self._list_theory_files()
        self._w_load_pick.options = opts
        if current in opts:
            self._w_load_pick.value = current

        target = self._w_load_pick.value
        self._status.clear_output()
        with self._status:
            if not target:
                print('[load] No file selected — pick one from the '
                      'dropdown first.')
                return
            path = os.path.join(self.theories_dir, target)
            try:
                spec = load_spec_from_file(path)
                self.load(spec)
            except Exception as exc:
                print(self._humanize_error(exc, 'preview'))
                return
            self._dirty = False
            try:
                self._chk_confirm_reset.value = False
                self._chk_confirm_load.value = False
                self._refresh_validation()
                self._refresh_save_hint()
            except Exception:
                pass
            print(f'[OK] Loaded {target}.  Edit the tabs and Save to '
                  f'write back.')

    def load(self, spec_or_path) -> None:
        """Populate the form from a spec dict OR a ``.theory.py`` path.

        Existing widget contents are cleared and replaced.  Convenient
        for programmatic round-trips (load + tweak + save) without
        going through the dropdown UI.
        """
        if isinstance(spec_or_path, str):
            spec = load_spec_from_file(spec_or_path)
        else:
            spec = dict(spec_or_path)

        # Top-of-form fields.
        self._w_name.value = spec.get('name', '') or ''
        self._w_description.value = spec.get('description', '') or ''

        # Populations.
        self._tbl_populations.clear()
        for p in spec.get('populations', []) or []:
            self._tbl_populations.add_row(values={
                'name':        p.get('name', ''),
                'size':        int(p.get('size', 1)),
                'description': p.get('description', '') or '',
            })
        # If no populations declared, leave the table empty so user
        # can add them; the dropdown providers will refresh themselves.

        # Refresh all population-aware dropdowns on dependent tabs
        # now that the Populations table reflects the new state.
        self._tbl_physical.refresh_dropdown_options('population')
        self._tbl_parameters.refresh_dropdown_options('index_1')
        self._tbl_parameters.refresh_dropdown_options('index_2')
        self._tbl_kernels.refresh_dropdown_options('index_1')
        self._tbl_kernels.refresh_dropdown_options('index_2')
        self._tbl_functions.refresh_dropdown_options('population')

        # Physical fields.
        self._tbl_physical.clear()
        # Reset the latex carry-through; repopulate from spec.
        self._loaded_extras['field_latex'] = {}
        for f in spec.get('physical_fields', []) or []:
            # Strip the auto-prefix 'd' if the loaded model went
            # through TheoryBuilder.physical_field with natural_name
            # — the user typed 'n' and the framework stored 'dn'.
            display_name = f.get('natural_name') or f.get('name', '')
            latex_str = (f.get('latex') or '').strip()
            if latex_str:
                self._loaded_extras['field_latex'][display_name] = latex_str
            self._tbl_physical.add_row(values={
                'name':        display_name,
                'population':  f.get('population') or '',
                # latex column removed from the UI (2026-05-27); the
                # string is kept on ``_loaded_extras`` and re-injected
                # by ``_collect()``.
                'description': f.get('description', '') or '',
            })

        # Parameters.  Map ``indexed_by`` back to (index_1, index_2)
        # dropdown selections, and ``default`` back to its text form.
        _NONE = '—'
        self._tbl_parameters.clear()
        # Auto-skip saddle parameters that the framework re-creates
        # from physical_field declarations (their names are
        # ``<natural>star`` and they carry mean_field=True).
        nat_names = {(f.get('natural_name') or f.get('name', ''))
                     for f in spec.get('physical_fields', []) or []}
        auto_saddle_names = {f'{n}star' for n in nat_names if n}
        for p in spec.get('parameters', []) or []:
            if p.get('mean_field') or p.get('name') in auto_saddle_names:
                continue
            ib = p.get('indexed_by') or []
            row = {
                'name':    p.get('name', ''),
                'index_1': (ib[0] if len(ib) >= 1 else _NONE),
                'index_2': (ib[1] if len(ib) >= 2 else _NONE),
                'domain':  p.get('domain', '') or '',
            }
            d = p.get('default')
            if d is not None:
                row['default'] = repr(d) if not isinstance(d, str) else d
            else:
                row['default'] = ''
            row['description'] = p.get('description', '') or ''
            self._tbl_parameters.add_row(values=row)

        # Functions.  ``args`` round-trips through a comma-joined
        # string back into the UI's text field.
        self._tbl_functions.clear()
        self._loaded_extras['function_latex'] = {}
        for fn in spec.get('functions', []) or []:
            args = fn.get('args') or []
            if isinstance(args, str):
                args_text = args
            else:
                args_text = ', '.join(args)
            fname = fn.get('name', '')
            latex_str = (fn.get('latex') or '').strip()
            if latex_str:
                self._loaded_extras['function_latex'][fname] = latex_str
            self._tbl_functions.add_row(values={
                'name':        fname,
                'population':  fn.get('population') or _NONE,
                'args':        args_text,
                'expression':  fn.get('expression', '') or '',
                # latex column removed from the UI (2026-05-27); see
                # _loaded_extras above.
                'description': fn.get('description', '') or '',
            })

        # Kernels.
        self._tbl_kernels.clear()
        self._loaded_extras['kernel_latex'] = {}
        for k in spec.get('kernels', []) or []:
            ib = k.get('indexed_by') or []
            kname = k.get('name', '')
            latex_str = (k.get('latex_name') or '').strip()
            if latex_str:
                self._loaded_extras['kernel_latex'][kname] = latex_str
            self._tbl_kernels.add_row(values={
                'name':       kname,
                'index_1':    (ib[0] if len(ib) >= 1 else _NONE),
                'index_2':    (ib[1] if len(ib) >= 2 else _NONE),
                'time_expr':  k.get('time_expr', '') or '',
                'freq_image': k.get('freq_image', '') or '',
                # latex_name column removed from the UI (2026-05-27);
                # see _loaded_extras above.
            })

        # CGFs.  Each row's response_legs cell accepts either a single
        # field name (legacy single-field cumulant) or a comma-separated
        # list (cross-field cumulant).  Loading round-trip: prefer the
        # explicit response_legs, fall back to legacy response_field.
        self._tbl_cgfs.clear()
        for c in spec.get('cgf_terms', []) or []:
            legs = c.get('response_legs')
            if isinstance(legs, (list, tuple)):
                legs_text = ', '.join(str(x) for x in legs)
            elif isinstance(legs, str) and legs.strip():
                legs_text = legs.strip()
            else:
                legs_text = c.get('response_field', '') or ''
            self._tbl_cgfs.add_row(values={
                'name':           c.get('name', ''),
                'response_legs':  legs_text,
                'order':          int(c.get('order', 2)),
                'coefficient':    c.get('coefficient', '') or '',
                'kernel':         c.get('kernel', '') or '',
            })

        # Action.
        self._w_action._text_w.value = spec.get('action_text', '') or ''

        # MF equations.
        # The new DAE form is ``spec['equations']`` — list of
        # ``{lhs, rhs, population}`` records.  Legacy ``mf_equations``
        # (list of ``{saddle, rhs}``) is auto-converted to the new
        # form: ``lhs = '<natural>[i]'`` (saddle name minus trailing
        # ``star``), population back-looked-up from the physical field
        # of that natural name.  Theories with both keys prefer the
        # explicit ``equations`` list.
        self._tbl_mfeqs.clear()
        equations = spec.get('equations') or []
        if not equations:
            # Back-compat: convert legacy ``set_mf_equation(saddle, rhs)``
            # entries.
            phys_pop_by_natural = {
                (f.get('natural_name') or f['name']): f.get('population')
                for f in (spec.get('physical_fields') or [])
            }
            for eq in spec.get('mf_equations', []) or []:
                saddle = (eq.get('saddle') or '').strip()
                natural = (saddle[:-4] if saddle.endswith('star')
                           else saddle)
                pop = phys_pop_by_natural.get(natural)
                equations.append({
                    'lhs':        f'{natural}[i]' if natural else '',
                    'rhs':        eq.get('rhs', ''),
                    'population': pop,
                })
        for eq in equations:
            pop = eq.get('population')
            self._tbl_mfeqs.add_row(values={
                'lhs':        eq.get('lhs', ''),
                'rhs':        eq.get('rhs', ''),
                'population': (_NONE if pop in (None, '', _NONE) else pop),
            })

        # Metadata text area.  Render the dict back as Python source
        # so the user can edit it.  The legacy ``default_fundamental``
        # spec key, if present in a loaded file, is now ignored — the
        # per-parameter ``default=...`` on the Parameters tab carries
        # the suggested values.
        # Stability-analysis toggle: restored from the spec's
        # ``stability_analysis`` key (set by load_spec_from_file when
        # parsing ``.stability_analysis(True)``).  Default False so
        # specs without the call show an unchecked box.
        self._w_stability_on.value = bool(spec.get('stability_analysis', False))

        md = spec.get('metadata') or {}
        if isinstance(md, dict):
            # Drive every structured Defaults widget from the loaded
            # metadata dict, with safe defaults for any missing key.
            self._w_k_default.value = int(md.get('k_default', 2) or 2)
            self._w_ell_default.value = int(md.get('ell_default', 0) or 0)
            self._w_tau_max.value = float(md.get('tau_max', 50.0) or 50.0)
            self._w_tau_step.value = float(md.get('tau_step', 0.5) or 0.5)

            self._tbl_ext_fields.clear()
            ext = md.get('recommended_external_fields') or []
            for entry in ext:
                # Tolerate both ('n', 1) tuples and ['n', 1] lists.
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    name, idx = entry[0], entry[1]
                    self._tbl_ext_fields.add_row(values={
                        'field': str(name),
                        'leaf_index': int(idx),
                    })

            self._w_fpi_default.value = int(
                md.get('fixed_point_index_default', 0) or 0)
            sb = md.get('seed_box_default', None)
            if sb is None:
                self._w_seed_box.value = ''
            elif isinstance(sb, dict):
                self._w_seed_box.value = repr(sb)
            else:
                self._w_seed_box.value = str(sb)

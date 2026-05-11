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
from IPython.display import display

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
)


# Default theories directory — relative to current working dir, since
# notebooks/ runs from there
_DEFAULT_THEORIES_DIR = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'theories')
)


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
            W.HTML('<h4>Theory metadata</h4>'),
            self._w_name, self._w_description,
            W.HTML('<p style="color:#555;font-size:90%;">'
                   'Number of populations and their sizes are declared on '
                   'the next tab (<b>Populations</b>).</p>'),
        ])

        # Tab 2: Populations — declare named populations + their sizes.
        # Other tabs read the current population list to populate
        # dropdowns: a field belongs to one population; a parameter /
        # kernel is indexed by zero / one / two populations.
        self._tbl_populations = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'E',    'width': '120px'},
                {'name': 'size',        'kind': 'int',
                 'default': 1,          'width': '80px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': 'excitatory population',
                 'width': '320px'},
            ],
            initial=[
                {'name': 'pop1', 'size': 1, 'description': ''},
                {'name': 'pop2', 'size': 1, 'description': ''},
            ],
        )
        tab_populations = W.VBox([
            W.HTML('<h4>Populations</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Declare each population by name and size.  Population "
                   "names are user-chosen identifiers (e.g. <code>E</code>, "
                   "<code>I</code>, or just <code>pop1</code>, <code>pop2</code>).  "
                   "Sizes must be positive integers — they determine the "
                   "shape of any field, parameter, or kernel indexed by "
                   "that population.  Subsequent tabs auto-update their "
                   "population dropdowns whenever you add / remove rows here."
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
                 'placeholder': 'n_E',  'width': '120px'},
                {'name': 'population',  'kind': 'select',
                 'options_provider': self._pop_names,
                 'width': '120px'},
                {'name': 'latex',       'kind': 'text',
                 'placeholder': '(auto: \\delta n_E)', 'width': '140px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': '',     'width': '280px'},
            ],
            initial=[
                {'name': 'n', 'population': 'pop1',
                 'description': 'firing rate'},
                {'name': 'v', 'population': 'pop1',
                 'description': 'voltage'},
            ],
        )
        tab_fields = W.VBox([
            W.HTML(
                '<h4>Physical fields</h4>'
                '<p style="color:#555;font-size:90%;">'
                'Each field is a vector of length <code>size(population)</code>. '
                'Pick a name (any identifier) and the population it belongs to.  '
                'The framework automatically creates:'
                '<ul style="margin-top:2px;">'
                '<li>the fluctuation field <code>d&lt;name&gt;</code> '
                '(used in the action — write <code>n_E[i]</code> or '
                '<code>dn_E[i]</code>; both refer to the fluctuation)</li>'
                '<li>the conjugate response field '
                '<code>&lt;name&gt;t</code> — MSR-pairing factor</li>'
                '<li>the saddle parameter <code>&lt;name&gt;star</code> '
                '— solved numerically by the saddle solver</li>'
                '</ul>'
                'Index range in the action follows the chosen population: '
                '<code>for i in pop_&lt;pop_name&gt;</code>.'
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
                 'placeholder': 'tau', 'width': '110px'},
                {'name': 'index_1',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '100px'},
                {'name': 'index_2',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '100px'},
                {'name': 'default',  'kind': 'text',
                 'placeholder': '10.0  /  [.., ..]  /  [[..]]',
                 'width': '340px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': '', 'width': '220px'},
            ],
            initial=[
                {'name': 'tau', 'index_1': _NONE, 'index_2': _NONE,
                 'default': '10.0'},
            ],
        )
        tab_params = W.VBox([
            W.HTML(
                '<h4>Parameters</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Numerical parameters.  Pick zero / one / two populations "
                "via the <code>index_1</code> and <code>index_2</code> dropdowns:"
                '<ul style="margin-top:2px;">'
                '<li><b>scalar</b>: both blank — e.g. <code>tau = 10.0</code></li>'
                '<li><b>vector</b>: <code>index_1</code> set — shape '
                '<code>(size(index_1),)</code></li>'
                '<li><b>matrix</b>: both set — shape '
                '<code>(size(index_1), size(index_2))</code> '
                '(row-first: outer index runs over <code>index_1</code>)</li>'
                '</ul>'
                'The <code>default</code> cell expects a Python literal of '
                "matching shape.  Changing an index dropdown updates the "
                "cell's placeholder to show the expected shape; if the "
                "<code>default</code> is empty, a template (e.g. "
                "<code>[[, , ], [, , ]]</code>) is auto-inserted."
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
                 'placeholder': 'phi',  'width': '100px'},
                {'name': 'population',  'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'args',        'kind': 'text',
                 'placeholder': 'v  (or "v, n" for multi-arg)',
                 'width': '140px'},
                {'name': 'expression',  'kind': 'text',
                 'placeholder': 'a[i] * v^2  /  a * v  /  ...',
                 'width': '260px'},
                {'name': 'latex',       'kind': 'text',
                 'placeholder': r'\varphi', 'width': '80px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': 'transfer function', 'width': '180px'},
            ],
            initial=[
                {'name': 'phi', 'population': _NONE,
                 'args': 'v', 'expression': 'a * v',
                 'latex': r'\varphi', 'description': 'transfer function'},
            ],
        )
        tab_functions = W.VBox([
            W.HTML(
                '<h4>Functions of field variables</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Declare any function of field variables.  Name them "
                "anything you like — <code>phi</code>, <code>f</code>, "
                "<code>response</code>, <code>kernel_response</code>, …  "
                "There is no reserved name."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "<b>population</b>: pick the population this function is "
                "bound to (or leave blank for a global function shared "
                "across all populations).  In the action, write "
                "<code>phi[i](v_E[i])</code> — the index <code>i</code> "
                "ranges over <code>pop_&lt;population&gt;</code>, the "
                "function's expression is evaluated with the population "
                "index <code>i</code> in scope (so it may reference "
                "<code>a[i]</code> for an indexed parameter), and the "
                "framework auto-Taylor-expands around the saddle of the "
                "function's argument."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                "<code>args</code> = comma-separated field-variable "
                "names (one per arg).  The first arg's natural name "
                "determines the saddle the function is expanded around: "
                "for <code>f(v)</code>, that's <code>vstar</code>."
                '</p>'),
            self._tbl_functions.show(),
        ])

        # Tab 6: Kernels
        # Same index_1 / index_2 pattern as Parameters.
        self._tbl_kernels = DynamicTable(
            columns=[
                {'name': 'name',       'kind': 'text',
                 'placeholder': 'g',   'width': '100px'},
                {'name': 'index_1',    'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'index_2',    'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE,     'width': '100px'},
                {'name': 'time_expr',  'kind': 'text',
                 'placeholder': '(1/tau_g)*exp(-t/tau_g)*heaviside(t)',
                 'width': '280px'},
                {'name': 'freq_image', 'kind': 'text',
                 'placeholder': '1/(1+I*omega*tau_g)',
                 'width': '220px'},
                {'name': 'latex_name', 'kind': 'text',
                 'placeholder': 'g',   'width': '80px'},
            ],
            initial=[
                {'name': 'g', 'index_1': _NONE, 'index_2': _NONE,
                 'freq_image': '1/(1+I*omega*tau_g)',
                 'latex_name': 'g'},
            ],
        )
        tab_kernels = W.VBox([
            W.HTML('<h4>Convolution kernels</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Each kernel must integrate to 1.  Provide either "
                   "<code>time_expr</code> (in <code>t</code> + parameters) or "
                   "<code>freq_image</code> (in <code>omega</code> + parameters). "
                   "<code>freq_image</code> is preferred — used directly by "
                   "the propagator builder."
                   '</p>'
                   '<p style="color:#555;font-size:90%;">'
                   "Index dropdowns work like in the Parameters tab:"
                   "<ul style='margin-top:4px;'>"
                   "<li><b>both blank</b>: shared scalar kernel — "
                   "use <code>g</code> in the action.</li>"
                   "<li><b>one set</b>: per-population kernel — use "
                   "<code>g[i]</code> in the action; the expression may "
                   "reference <code>i</code> and any indexed parameter "
                   "(<code>tau_g[i]</code>, etc.).</li>"
                   "<li><b>both set</b>: per-pair kernel — use "
                   "<code>g[i, j]</code> in the action; the expression "
                   "may reference <code>i, j</code> and matrix-indexed "
                   "parameters like <code>tau_g[i, j]</code>.</li>"
                   "</ul>"
                   '</p>'),
            self._tbl_kernels.show(),
        ])

        # Tab 6: Non-closed-form CGFs
        self._tbl_cgfs = DynamicTable(
            columns=[
                {'name': 'name',           'kind': 'text',
                 'placeholder': 'X',       'width': '60px'},
                {'name': 'response_field', 'kind': 'text',
                 'placeholder': 'mt',      'width': '90px'},
                {'name': 'order',          'kind': 'int',
                 'default': 2,             'width': '60px'},
                {'name': 'coefficient',    'kind': 'text',
                 'placeholder': 'lambda_X * p_part * (...)',
                 'width': '300px'},
                {'name': 'kernel',         'kind': 'text',
                 'placeholder': '(optional time-domain factor)',
                 'width': '220px'},
            ],
            initial=[],
        )
        tab_cgfs = W.VBox([
            W.HTML('<h4>Non-closed-form CGF terms</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "For external noises declared via cumulants (GTaS, etc.). "
                   "Multiple rows with the same <code>name</code>+<code>order</code> "
                   "sum into one cumulant.  Leave empty for closed-form-only theories."
                   '</p>'),
            self._tbl_cgfs.show(),
        ])

        # Tab 7: Action — full S in physical fields, with explicit sums.
        self._w_action = textarea_input(
            'Action S',
            placeholder=(
                "sum(\n"
                "    nt[i] * n[i]\n"
                "    - (exp(nt[i]) - 1) * phi[i](v[i])\n"
                "    + vt[i] * (\n"
                "        (tau * Dt + 1) * v[i]\n"
                "        - E[i]\n"
                "        - sum(w[i, j] * g * n[j] for j in pop)\n"
                "    )\n"
                "    for i in pop\n"
                ")"
            ),
            rows=14,
            width='820px',
        )
        tab_action = W.VBox([
            W.HTML(
                '<h4>Action S</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Write the <strong>full</strong> action as a single Sage "
                "expression in physical observables (<code>n[i]</code>, "
                "<code>v[i]</code>, …).  All sums over populations are "
                "explicit — use Python comprehension syntax against the "
                "pre-bound <code>pop = range(N_populations)</code>:"
                "<pre style='background:#f5f5f5;padding:8px;'>"
                "sum(... for i in pop)\n"
                "sum(... for j in pop)\n"
                "sum(... for i in pop for j in pop)</pre>"
                "Terms that don't iterate over <code>pop</code> "
                "(e.g. couplings to a different external population set) "
                "go outside any sum and add normally."
                '</p>'
                '<p style="color:#555;font-size:90%;">'
                'Conventions:'
                '<ul style="margin-top:2px;">'
                '<li><strong>Use physical observables</strong>: write '
                '<code>n[i]</code>, <code>v[i]</code> — the framework '
                'expands these to <code>nstar[i] + dn[i]</code>, '
                '<code>vstar[i] + dv[i]</code> behind the scenes.  '
                'The user does not type <code>nstar</code>, <code>dn</code>, '
                'or any saddle-fluctuation decomposition.</li>'
                '<li><strong>Transfer functions are indexed</strong>: '
                'write <code>phi[i](v[i])</code>.  Pass the full physical '
                'field as the argument; the framework Taylor-expands '
                'around the saddle and substitutes '
                '<code>phi[i](vstar[i]) → nstar[i]</code> automatically.</li>'
                '<li><strong>Matrix subscript</strong>: '
                '<code>w[i, j]</code> (tuple) or <code>w[i][j]</code> '
                '(chained) — both work.</li>'
                '<li><strong>Differential operators</strong>: '
                '<code>Dt</code> is the time derivative ∂<sub>t</sub>. '
                'Compose as a regular Sage operator: '
                '<code>(tau * Dt + 1) * v[i]</code>.</li>'
                '</ul>'
                '</p>'),
            self._w_action,
        ])

        # Tab 8: Mean-field equations
        self._tbl_mfeqs = DynamicTable(
            columns=[
                {'name': 'saddle', 'kind': 'text',
                 'placeholder': 'vstar / nstar / mstar', 'width': '140px'},
                {'name': 'rhs',    'kind': 'text',
                 'placeholder': 'E[i] + sum(w[i, j] * g * nstar[j] for j in pop)',
                 'width': '480px'},
            ],
            initial=[
                {'saddle': 'vstar',
                 'rhs': 'E[i] + sum(w[i, j] * g * nstar[j] for j in pop)'},
                {'saddle': 'nstar', 'rhs': 'phi(vstar[i])'},
            ],
        )
        tab_mfeqs = W.VBox([
            W.HTML(
                '<h4>Mean-field equations</h4>'
                '<p style="color:#555;font-size:90%;">'
                "One row per saddle quantity.  Per-<code>i</code> form. "
                "The framework iterates on <code>nstar</code> numerically — "
                "its equation (<code>nstar[i] = phi(vstar[i])</code>) is the "
                "self-consistency closure but is <strong>not</strong> substituted "
                "into the action.  Other saddles (<code>vstar</code>, "
                "<code>mstar</code>) are substituted into the action via the "
                "<code>mf_bg_conditions</code> hook."
                '</p>'),
            self._tbl_mfeqs.show(),
        ])

        # Tab 9: Defaults
        self._w_def_fund = W.Textarea(
            value=(
                "{\n"
                "    'E':     [0.78, 0.81],\n"
                "    'w':     [[0.30, 0.25], [0.30, 0.35]],\n"
                "    'tau':   10.0,\n"
                "    'a':     1.0,\n"
                "    'tau_g': 2.5,\n"
                "}"
            ),
            placeholder='Python dict literal',
            layout=W.Layout(width='540px', height='150px'),
        )
        self._w_metadata = W.Textarea(
            value=(
                "{\n"
                "    'k_default':                   2,\n"
                "    'ell_default':                 0,\n"
                "    'recommended_external_fields': [('n', 1), ('n', 2)],\n"
                "    'tau_max':                     50.0,\n"
                "    'tau_step':                    0.5,\n"
                "}"
            ),
            placeholder='Python dict literal',
            layout=W.Layout(width='540px', height='150px'),
        )
        tab_defaults = W.VBox([
            W.HTML('<h4>Default fundamental parameter values</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Suggested numerical values for the parameters declared "
                   "above.  The runner uses these by default but they can be "
                   "overridden per-run via <code>FUNDAMENTAL_OVERRIDE</code>."
                   '</p>'),
            self._w_def_fund,
            W.HTML('<br><h4>Run metadata</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Suggestions for k, max_ell, external_fields, τ-grid extent. "
                   "The runner uses these as defaults."
                   '</p>'),
            self._w_metadata,
        ])

        # Compose into Tab widget
        self._tabs = W.Tab(children=[
            tab_model, tab_populations, tab_fields, tab_params,
            tab_functions, tab_kernels, tab_cgfs, tab_action,
            tab_mfeqs, tab_defaults,
        ])
        for i, title in enumerate([
            '1. Model', '2. Populations', '3. Fields', '4. Parameters',
            '5. Functions', '6. Kernels', '7. CGFs', '8. Action',
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
        self._btn_reset   = W.Button(description='Reset',
                                     button_style='warning',
                                     layout=W.Layout(width='100px'))
        self._btn_preview.on_click(self._on_preview)
        self._btn_save.on_click(self._on_save)
        self._btn_reset.on_click(self._on_reset)

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
            '<code>notebooks/theory_runner.ipynb</code> picks it up automatically.'
            '</p>')

        self._root = W.VBox([
            self._header,
            self._tabs,
            W.HBox([self._w_save_path, self._btn_save,
                    self._btn_preview, self._btn_reset]),
            self._status,
        ])

    # ── Public API ────────────────────────────────────────────────
    def show(self) -> None:
        """Render the form in the current notebook cell."""
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
            for k in ('latex', 'description'):
                v = (r.get(k) or '').strip()
                if v:
                    entry[k] = v
            physical_fields.append(entry)

        # Parameters — name + indexed_by list (from dropdowns) + default.
        params = []
        for row in self._tbl_parameters.get_rows():
            name = (row.get('name') or '').strip()
            if not name:
                continue
            entry = {'name': name}
            idx = _index_list(row)
            if idx:
                entry['indexed_by'] = idx
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
            for k in ('latex', 'description'):
                v = row.get(k)
                if v:
                    entry[k] = v
            functions.append(entry)

        # Kernels — name + indexed_by + time_expr/freq_image + latex.
        kernels = []
        for row in self._tbl_kernels.get_rows():
            name = (row.get('name') or '').strip()
            if not name:
                continue
            entry = {'name': name}
            idx = _index_list(row)
            if idx:
                entry['indexed_by'] = idx
            for k in ('time_expr', 'freq_image', 'latex_name'):
                v = (row.get(k) or '').strip()
                if v:
                    entry[k] = v
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
            'cgf_terms':       self._tbl_cgfs.get_rows(),
            'action_text':     self._w_action.get_value(),
            'mf_equations':    [
                {'saddle': r['saddle'], 'rhs': r['rhs']}
                for r in self._tbl_mfeqs.get_rows()
            ],
            'default_fundamental': _eval_dict(self._w_def_fund.value,
                                              'default_fundamental'),
            'metadata':            _eval_dict(self._w_metadata.value,
                                              'metadata'),
        }

    # ── Button handlers ───────────────────────────────────────────
    def _on_preview(self, _btn) -> None:
        self._status.clear_output()
        with self._status:
            try:
                spec = self._collect()
                src = render_theory_file(spec)
                print(src)
            except Exception as e:
                print(f'[ERROR] {e}')

    def _on_save(self, _btn) -> None:
        self._status.clear_output()
        with self._status:
            try:
                spec = self._collect()
                target = (self._w_save_path.value.strip()
                          or self.theories_dir)
                if not os.path.isabs(target) and not target.endswith('.py'):
                    target = os.path.join(self.theories_dir, target)
                path = save_theory_to_file(spec, target)
                self.last_saved = path
                print(f'[OK] Wrote {path}')
                print(f'\nIn notebooks/theory_runner.ipynb set:')
                rel = os.path.basename(path)[:-len('.theory.py')]
                print(f'    THEORY_NAME = {rel!r}')
            except Exception as e:
                import traceback
                traceback.print_exc()

    def _on_reset(self, _btn) -> None:
        # Re-build the whole UI to wipe state cleanly.
        self._build_widgets()
        self.show()

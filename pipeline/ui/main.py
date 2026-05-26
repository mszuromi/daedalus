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
    load_spec_from_file,
)


# Default theories directory — relative to current working dir, since
# notebooks/ runs from there
_DEFAULT_THEORIES_DIR = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'theories')
)


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
                {'name': 'domain',   'kind': 'select',
                 'options': ['', 'positive', 'real'],
                 'default': '', 'width': '90px'},
                {'name': 'default',  'kind': 'text',
                 'placeholder': '10.0  /  [.., ..]  /  [[..]]',
                 'width': '300px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': '', 'width': '180px'},
            ],
            initial=[
                {'name': 'tau', 'index_1': _NONE, 'index_2': _NONE,
                 'domain': 'positive', 'default': '10.0'},
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

        # Tab 6: Non-closed-form CGFs (correlated noise cumulants)
        self._tbl_cgfs = DynamicTable(
            columns=[
                {'name': 'name',           'kind': 'text',
                 'placeholder': 'X',       'width': '60px'},
                {'name': 'response_legs',  'kind': 'text',
                 'placeholder': 'mt   or   xt, yt',
                 'width': '160px'},
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
            W.HTML(
                '<h4>Non-closed-form CGF terms (correlated noise cumulants)</h4>'
                '<p style="color:#555;font-size:90%;">'
                "One row per (name, order, leg-tuple) cumulant contribution.  "
                "Multiple rows with the same "
                "<code>(name, order, response_legs)</code> sum into one entry; "
                "rows with the same <code>name</code> but different "
                "<code>response_legs</code> stay distinct.  Leave empty for "
                "closed-form-only theories.<br><br>"
                "<b>response_legs</b>: one response-field name per cumulant "
                "leg (length = <code>order</code>).  Comma-separated: "
                "<code>mt</code> (single, broadcast to all legs &mdash; "
                "legacy GTaS path) or <code>xt, yt</code> for cross-field "
                "cumulants (e.g. cross-correlated noise on 2D OU).  "
                "Single-entry rows are equivalent to the legacy "
                "<code>response_field</code> column.<br><br>"
                "<b>coefficient</b>: the cumulant kernel's amplitude at "
                "&tau;=0.  For Gaussian white noise with diffusion <code>D</code>, "
                "use <code>2*D</code> (standard MSR convention "
                "&kappa;<sup>(2)</sup> = 2D&middot;&delta;).<br><br>"
                "<b>kernel</b>: optional &tau;-dependent factor.  "
                "<code>dirac_delta(tau)</code> for white, "
                "<code>exp(-abs(tau)/tauc)/(2*tauc)</code> for OU-colored, "
                "<code>dirac_delta(t1)*dirac_delta(t2)</code> for "
                "shot-noise &kappa;<sup>(3)</sup>, etc.  At order n use "
                "<code>tau</code> (n=2) or <code>t1, t2, ...</code> "
                "(n&ge;3) as the relative-time variables."
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

        # Tab 8: Mean-field equations — DAE form.
        # One row per equation (residual ``LHS - RHS = 0``).  Dt allowed
        # on the LHS; absent ⇒ algebraic.  At MF the DAE solver sets
        # Dt → 0; for stability it keeps Dt symbolic.  Population
        # determines the index range for ``i`` on both sides.
        self._tbl_mfeqs = DynamicTable(
            columns=[
                {'name': 'lhs',        'kind': 'text',
                 'placeholder': '(tau[i]*Dt + 1) * v[i]',
                 'width': '240px'},
                {'name': 'rhs',        'kind': 'text',
                 'placeholder': 'E[i] + sum(w[i, j]*n[j] for j in E)',
                 'width': '360px'},
                {'name': 'population', 'kind': 'select',
                 'options_provider': _pop_opts_with_none,
                 'default': _NONE, 'width': '110px'},
            ],
            initial=[
                {'lhs':        '(tau[i]*Dt + 1) * v[i]',
                 'rhs':        'E[i] + sum(w[i, j]*n[j] for j in E)',
                 'population': _NONE},
                {'lhs':        'n[i]',
                 'rhs':        'phi[i](v[i])',
                 'population': _NONE},
            ],
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
                         "{'n': (-1.5, 1.5), 'v': (-3.0, 3.0)}"),
            layout=W.Layout(width='460px', height='60px'),
        )

        tab_mfeqs = W.VBox([
            W.HTML(
                '<h4>Mean-field equations (DAE form)</h4>'
                '<p style="color:#555;font-size:90%;">'
                "One row per equation — a residual <code>LHS &minus; RHS = 0</code>.  "
                "Both sides are Sage-syntax strings in <code>i</code> "
                "(population index), declared parameters / kernels, "
                "the <code>Dt</code> time-derivative operator, and "
                "state-variable fields.  At mean field the solver sets "
                "<code>Dt &rarr; 0</code> to get an algebraic system and "
                "runs multi-start Newton over every distinct root; the "
                "linear-stability check uses the same equations with "
                "<code>Dt</code> kept symbolic.  "
                "Pass <code>population</code> to expand <code>[i]</code> "
                "over that population's index range; leave as "
                f"<code>{_NONE}</code> for a scalar equation.  "
                "Equations are declaration-ordered and treated as one "
                "system; ordering doesn't affect the solver.  "
                "<strong>No <code>Conv(...)</code> on the RHS</strong> &mdash; "
                "stationary MF assumption requires the user to "
                "pre-collapse normalized-kernel convolutions of constants."
                '</p>'),
            self._tbl_mfeqs.show(),
            W.HTML(
                '<br><h4>Multi-root selection</h4>'
                '<p style="color:#555;font-size:90%;">'
                "When the DAE has multiple fixed points (e.g. a bistable "
                "theory), the solver finds all of them and sorts the "
                "list ascending by the first declared physical field's "
                "first population index.  <code>fixed_point_index = 0</code> "
                "picks the lowest, <code>1</code> the next, and so on. "
                "This widget sets the THEORY's default; runs "
                "can override via "
                "<code>compute_cumulants(..., fixed_point_index=N)</code>."
                '<br><br>'
                "<strong>Linear-stability filtering</strong> (the box below) "
                "is OFF by default. When ON, the solver classifies every "
                "converged root via the generalized-eigenvalue Jacobian, "
                "filters down to the LINEARLY STABLE subset, and routes "
                "<code>fixed_point_index</code> over those (the "
                "diagrammatic series isn't defined around an unstable "
                "saddle).  Unstable roots stay inspectable via "
                "<code>th['mf_unstable_roots']</code>.  "
                "Leave OFF when your equations are all algebraic "
                "(voltages integrated out, no <code>Dt</code> anywhere) "
                "&mdash; with <code>A &equiv; 0</code> in the "
                "<code>(&sigma;A + B)</code> eigenproblem there's nothing "
                "to score, and the filter would either be vacuous or, "
                "in degenerate cases, mis-classify the saddle."
                '</p>'),
            self._w_fpi_default,
            self._w_stability_on,
            W.HTML(
                '<br><h4>Initial-guess seed box (optional)</h4>'
                '<p style="color:#555;font-size:90%;">'
                "Per-state-variable sampling box for multi-start Newton, "
                "as a Python dict literal mapping variable name to "
                "<code>(low, high)</code>.  Leave blank for the default "
                "domain-aware box (positive&nbsp;&rarr;&nbsp;<code>[0, 5&middot;scale]</code>, "
                "real&nbsp;&rarr;&nbsp;<code>[&minus;3&middot;scale, 3&middot;scale]</code> "
                "where scale = max&nbsp;parameter&nbsp;magnitude).  Override only "
                "if the default box misses your roots (e.g. tanh-bounded "
                "states that need <code>(&minus;1.5, 1.5)</code>)."
                '</p>'),
            self._w_seed_box,
        ])

        # Tab 9: Defaults (run metadata only — default-fundamental
        # values are now sourced from each parameter's ``default=...``
        # declaration on the Parameters tab, so a separate "default
        # fundamental" textarea would be redundant and a foot-gun for
        # drift).
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
            W.HTML('<h4>Run metadata</h4>'
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

        self._root = W.VBox([
            self._header,
            self._tabs,
            W.HBox([self._w_save_path, self._btn_save,
                    self._btn_preview, self._btn_precompute,
                    self._btn_reset]),
            W.HBox([self._w_load_pick, self._btn_load]),
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
        """Build the METADATA dict, merging the user's Defaults-tab
        Python literal with the MF-tab widget values (default
        ``fixed_point_index`` + optional ``seed_box``).  The MF widgets
        WIN on conflict so the explicit per-tab controls take
        precedence over a stale literal."""
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

        md = _safe_dict(self._w_metadata.value, 'metadata') or {}
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
                print(f'[precompute] cannot collect spec: {e!r}')
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
                import traceback
                print(f'[precompute] model build failed:')
                traceback.print_exc()
                return

            print(f'[precompute] model: {model.get("name", "<unnamed>")!r}')
            try:
                result = precompute(model, verbose=True)
            except Exception as e:
                import traceback
                print(f'[precompute] runtime error:')
                traceback.print_exc()
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
        the form widgets from its spec."""
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
                print('[ERROR] No file selected.')
                return
            path = os.path.join(self.theories_dir, target)
            try:
                spec = load_spec_from_file(path)
                self.load(spec)
                print(f'[OK] Loaded {target}.  Edit the tabs and Save to '
                      f'write back.')
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f'[ERROR] Could not load {target}: {exc}')

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
        for f in spec.get('physical_fields', []) or []:
            # Strip the auto-prefix 'd' if the loaded model went
            # through TheoryBuilder.physical_field with natural_name
            # — the user typed 'n' and the framework stored 'dn'.
            display_name = f.get('natural_name') or f.get('name', '')
            self._tbl_physical.add_row(values={
                'name':        display_name,
                'population':  f.get('population') or '',
                'latex':       f.get('latex', '') or '',
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
        for fn in spec.get('functions', []) or []:
            args = fn.get('args') or []
            if isinstance(args, str):
                args_text = args
            else:
                args_text = ', '.join(args)
            self._tbl_functions.add_row(values={
                'name':        fn.get('name', ''),
                'population':  fn.get('population') or _NONE,
                'args':        args_text,
                'expression':  fn.get('expression', '') or '',
                'latex':       fn.get('latex', '') or '',
                'description': fn.get('description', '') or '',
            })

        # Kernels.
        self._tbl_kernels.clear()
        for k in spec.get('kernels', []) or []:
            ib = k.get('indexed_by') or []
            self._tbl_kernels.add_row(values={
                'name':       k.get('name', ''),
                'index_1':    (ib[0] if len(ib) >= 1 else _NONE),
                'index_2':    (ib[1] if len(ib) >= 2 else _NONE),
                'time_expr':  k.get('time_expr', '') or '',
                'freq_image': k.get('freq_image', '') or '',
                'latex_name': k.get('latex_name', '') or '',
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
            # Pull the MF-tab widget values out of metadata so the
            # textarea shows the "rest" while the dedicated widgets
            # carry their own values.
            md_for_textarea = dict(md)
            self._w_fpi_default.value = int(
                md_for_textarea.pop('fixed_point_index_default', 0) or 0)
            sb = md_for_textarea.pop('seed_box_default', None)
            if sb is None:
                self._w_seed_box.value = ''
            elif isinstance(sb, dict):
                self._w_seed_box.value = repr(sb)
            else:
                self._w_seed_box.value = str(sb)
            self._w_metadata.value = (repr(md_for_textarea)
                                      if md_for_textarea else '{}')

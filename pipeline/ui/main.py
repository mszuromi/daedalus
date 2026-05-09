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

    # ── Tab construction ──────────────────────────────────────────
    def _build_widgets(self) -> None:
        # Tab 1: Model — name + populations
        self._w_name = W.Text(
            value='My Stochastic Theory', placeholder='Theory name',
            layout=W.Layout(width='400px'),
            description='Name:', style={'description_width': '80px'},
        )
        self._w_npop = W.IntText(
            value=2, layout=W.Layout(width='80px'),
            description='Populations:',
            style={'description_width': '120px'},
        )
        self._w_description = W.Textarea(
            value='', placeholder='Optional theory description / notes',
            layout=W.Layout(width='600px', height='60px'),
            description='Description:',
            style={'description_width': '120px'},
        )
        tab_model = W.VBox([self._w_name, self._w_npop,
                            self._w_description])

        # Tab 2: Field variables
        self._tbl_response = DynamicTable(
            columns=[
                {'name': 'name',         'kind': 'text',
                 'placeholder': 'nt',    'width': '100px'},
                {'name': 'indexed',      'kind': 'bool',
                 'default': True,        'width': '60px'},
                {'name': 'latex',        'kind': 'text',
                 'placeholder': r'\tilde n', 'width': '100px'},
                {'name': 'description',  'kind': 'text',
                 'placeholder': 'response field', 'width': '240px'},
            ],
            initial=[
                {'name': 'nt', 'indexed': True, 'latex': r'\tilde n'},
                {'name': 'vt', 'indexed': True, 'latex': r'\tilde v'},
            ],
        )
        self._tbl_physical = DynamicTable(
            columns=[
                {'name': 'name',         'kind': 'text',
                 'placeholder': 'dn',    'width': '100px'},
                {'name': 'natural_name', 'kind': 'text',
                 'placeholder': 'n',     'width': '90px'},
                {'name': 'indexed',      'kind': 'bool',
                 'default': True,        'width': '60px'},
                {'name': 'latex',        'kind': 'text',
                 'placeholder': r'\delta\dot n', 'width': '100px'},
                {'name': 'description',  'kind': 'text',
                 'placeholder': 'fluctuation field', 'width': '200px'},
            ],
            initial=[
                {'name': 'dn', 'natural_name': 'n', 'indexed': True,
                 'latex': r'\delta\dot n'},
                {'name': 'dv', 'natural_name': 'v', 'indexed': True,
                 'latex': r'\delta v'},
            ],
        )
        tab_fields = W.VBox([
            W.HTML('<h4>Response fields</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   '"Tilde" fields conjugate to physical observables. '
                   'Typically one per physical field.'
                   '</p>'),
            self._tbl_response.show(),
            W.HTML('<br><h4>Physical fields</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   'Fluctuation variables (`dn`, `dv`, …) used in the action. '
                   '<code>natural_name</code> is the user-facing letter '
                   '(e.g. <code>n</code>) external_fields uses.'
                   '</p>'),
            self._tbl_physical.show(),
        ])

        # Tab 3: Parameters
        self._tbl_parameters = DynamicTable(
            columns=[
                {'name': 'name',         'kind': 'text',
                 'placeholder': 'tau',   'width': '90px'},
                {'name': 'type',         'kind': 'select',
                 'options': ['scalar', 'vector', 'matrix'],
                 'default': 'scalar',    'width': '90px'},
                {'name': 'default',      'kind': 'text',
                 'placeholder': '10.0 / [0.5, 0.5] / [[..]]',
                 'width': '180px'},
                {'name': 'mean_field',   'kind': 'bool',
                 'default': False,       'width': '80px'},
                {'name': 'natural_name', 'kind': 'text',
                 'placeholder': 'n / v / m', 'width': '90px'},
                {'name': 'domain',       'kind': 'text',
                 'placeholder': 'positive (or blank)', 'width': '130px'},
                {'name': 'description',  'kind': 'text',
                 'placeholder': '',       'width': '180px'},
            ],
            initial=[
                {'name': 'nstar', 'type': 'vector', 'mean_field': True,
                 'natural_name': 'n', 'domain': 'positive'},
                {'name': 'vstar', 'type': 'vector', 'mean_field': True,
                 'natural_name': 'v'},
                {'name': 'E',     'type': 'vector'},
                {'name': 'tau',   'type': 'scalar', 'default': '10.0',
                 'domain': 'positive'},
                {'name': 'a',     'type': 'scalar', 'default': '1.0'},
                {'name': 'tau_g', 'type': 'scalar', 'default': '2.5',
                 'domain': 'positive'},
                {'name': 'w',     'type': 'matrix'},
            ],
        )
        tab_params = W.VBox([
            W.HTML('<h4>Parameters</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Saddle quantities (n*, v*, …) need <code>mean_field=True</code> "
                   "and a <code>natural_name</code> so <code>mf['n', i]</code> works. "
                   '<code>default</code> is parsed by Python: <code>10.0</code> for '
                   'scalars, <code>[0.5, 0.5]</code> for vectors, '
                   '<code>[[0.3, 0.25], [0.3, 0.35]]</code> for matrices.'
                   '</p>'),
            self._tbl_parameters.show(),
        ])

        # Tab 4: Functions
        self._tbl_functions = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'phi',  'width': '80px'},
                {'name': 'args',        'kind': 'text',
                 'placeholder': 'v  (or "v, t" for multi-arg)',
                 'width': '160px'},
                {'name': 'expression',  'kind': 'text',
                 'placeholder': 'a*v^2 / 1/(1+exp(-v)) / ...',
                 'width': '280px'},
                {'name': 'latex',       'kind': 'text',
                 'placeholder': r'\varphi', 'width': '80px'},
                {'name': 'description', 'kind': 'text',
                 'placeholder': 'transfer function', 'width': '180px'},
            ],
            initial=[
                {'name': 'phi', 'args': 'v', 'expression': 'a * v',
                 'latex': r'\varphi', 'description': 'transfer function'},
            ],
        )
        tab_functions = W.VBox([
            W.HTML('<h4>Functions of field variables</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Function name <code>phi</code> is reserved as the "
                   "saddle-point transfer function (auto-Taylor-expanded). "
                   "Other functions are evaluated concretely. "
                   "<code>args</code> = comma-separated field-variable names."
                   '</p>'),
            self._tbl_functions.show(),
        ])

        # Tab 5: Kernels
        self._tbl_kernels = DynamicTable(
            columns=[
                {'name': 'name',        'kind': 'text',
                 'placeholder': 'g',    'width': '80px'},
                {'name': 'time_expr',   'kind': 'text',
                 'placeholder': '(1/tau_g)*exp(-t/tau_g)*heaviside(t)',
                 'width': '300px'},
                {'name': 'freq_image',  'kind': 'text',
                 'placeholder': '1/(1+I*omega*tau_g)',
                 'width': '220px'},
                {'name': 'latex_name',  'kind': 'text',
                 'placeholder': 'g',    'width': '80px'},
            ],
            initial=[
                {'name': 'g', 'freq_image': '1/(1+I*omega*tau_g)',
                 'latex_name': 'g'},
            ],
        )
        tab_kernels = W.VBox([
            W.HTML('<h4>Convolution kernels</h4>'
                   '<p style="color:#555;font-size:90%;">'
                   "Each kernel must integrate to 1.  Provide either "
                   "<code>time_expr</code> (in <code>t</code> + parameters) or "
                   "<code>freq_image</code> (in <code>omega</code> + parameters). "
                   "<code>freq_image</code> is preferred — it's used directly by "
                   "propagator construction."
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

        # Tab 7: Action
        self._w_action = textarea_input(
            'Action S_i (per-population integrand)',
            placeholder=("nt[i] * (nstar[i] + dn[i])\n"
                         "- (exp(nt[i]) - 1) * phi(dv[i])\n"
                         "+ vt[i] * ( (tau*Dt + 1) * dv[i]\n"
                         "           + vstar[i] - E[i]\n"
                         "           - sum(w[i, j] * g * (nstar[j] + dn[j]) for j in pop) )"),
            rows=12,
            width='760px',
        )
        tab_action = W.VBox([
            W.HTML(
                '<h4>Action S<sub>i</sub></h4>'
                '<p style="color:#555;font-size:90%;">'
                "Per-population integrand.  Framework wraps "
                "<code>sum(... for i in pop)</code>.  "
                "<code>i</code> is reserved as the implicit free index. "
                "Inner sums use Python comprehension syntax: "
                "<code>sum(w[i, j] * dn[j] for j in pop)</code>. "
                "Transfer-function arguments are <strong>fluctuations</strong>: "
                "write <code>phi(dv[i])</code>, not <code>phi(vstar[i] + dv[i])</code> — "
                "the framework auto-Taylor-expands and substitutes "
                "<code>phi(0) → nstar[i]</code>."
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
            tab_model, tab_fields, tab_params, tab_functions,
            tab_kernels, tab_cgfs, tab_action, tab_mfeqs, tab_defaults,
        ])
        for i, title in enumerate([
            '1. Model', '2. Fields', '3. Parameters', '4. Functions',
            '5. Kernels', '6. CGFs', '7. Action', '8. MF eqs', '9. Defaults',
        ]):
            self._tabs.set_title(i, title)

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

        # Parse parameter defaults — each is a string the user typed.
        params = []
        for row in self._tbl_parameters.get_rows():
            entry = {'name': row['name'], 'type': row.get('type', 'scalar')}
            d = (row.get('default') or '').strip()
            if d:
                try:
                    entry['default'] = eval(d, {'__builtins__': {}}, {})
                except Exception:
                    entry['default'] = d   # leave as string if unparseable
            for k in ('mean_field', 'natural_name', 'domain', 'description'):
                v = row.get(k)
                if v not in (None, '', False):
                    entry[k] = v
            params.append(entry)

        # Functions: split args by comma
        functions = []
        for row in self._tbl_functions.get_rows():
            args = [a.strip() for a in (row.get('args') or '').split(',')
                    if a.strip()]
            entry = {'name': row['name'], 'args': args,
                     'expression': row.get('expression', '')}
            for k in ('latex', 'description'):
                v = row.get(k)
                if v:
                    entry[k] = v
            functions.append(entry)

        kernels = []
        for row in self._tbl_kernels.get_rows():
            entry = {'name': row['name']}
            for k in ('time_expr', 'freq_image', 'latex_name'):
                v = row.get(k)
                if v:
                    entry[k] = v
            kernels.append(entry)

        return {
            'name':            self._w_name.value,
            'n_populations':   int(self._w_npop.value),
            'description':     self._w_description.value,
            'response_fields': self._tbl_response.get_rows(),
            'physical_fields': self._tbl_physical.get_rows(),
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

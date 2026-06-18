# Theory Builder UI (ipywidgets authoring front-end)

*Subsystem slug: `ui`*

Primary source files:

- `pipeline/ui/main.py` (3072 lines) — the `TheoryUI` form: the 10-tab editor, the
  validation/readiness sidebar, the Save / Preview / Pre-compute / Load / Reset / Open-in-runner
  buttons, and the `_collect()` method that snapshots the form into a *spec dict*.
- `pipeline/ui/widgets.py` (438 lines) — the small reusable ipywidgets primitives the tabs
  are composed from: `DynamicTable` (add/remove-row spreadsheet), `vector_input`,
  `matrix_input`, `expression_input`, `textarea_input`, `paste_button`,
  `read_system_clipboard`.

Supporting files read for grounding (not the primary subject, but the consumers/producers
on either side of the UI):

- `pipeline/ui/__init__.py` — re-exports `TheoryUI`.
- `pipeline/theory_serialize.py` (900 lines) — the module that *renders* the spec dict into
  a `.theory.py` file (`render_theory_file`, `save_theory_to_file`) and *reverses* it
  (`load_spec_from_file`). The UI is one giant producer/consumer of this module.
- `pipeline/_precompute.py` — `precompute(model, …)`, invoked by the UI's **Pre-compute**
  button.
- `theories/ou_quartic.theory.py` — a concrete example of the file the UI emits.

---

## Overview

### What this subsystem is for, in plain language

The MSR-JD diagrammatic pipeline in this repository computes connected cumulants
(correlation / response functions) of a stochastic field theory. To do anything, the
pipeline needs a *theory file* — a `theories/<name>.theory.py` module with a `build()`
function that constructs the theory via a fluent builder API (`TemporalTheoryBuilder(...)` /
`SpatialTheoryBuilder(...)`, chained `.physical_field(...)`, `.parameter(...)`,
`.set_action_text(...)`, `.equation(...)`, `.build()`). Hand-writing that builder chain
correctly requires knowing the framework's naming conventions (the auto-generated `<name>t`
response field, the `<name>star` saddle parameter), the action grammar (`Dt` as a
time-derivative operator, `for i in pop` comprehensions, `Laplacian` in spatial theories),
and a pile of optional knobs (boundary conditions, Dyson dressing order, operator-IR mode,
stability analysis, run metadata).

**This subsystem is the point-and-click front door that writes that file for you.** A user
opens `notebooks/theory_builder.ipynb`, runs one cell (`from pipeline.ui import TheoryUI;
TheoryUI().show()`), and gets a tabbed form rendered *inline in the notebook* — built
entirely from [ipywidgets](#external-tools-used) (the Jupyter interactive-widget library).
The user fills in tabs (Model / Populations / Fields / Parameters / Functions / Kernels /
Noise / Action / Mean-field / Defaults), watches a live **readiness sidebar** flag undeclared
names and reserved-symbol collisions, and clicks **Save theory file**. The UI snapshots the
form into a *spec dict* and hands it to `pipeline.theory_serialize.save_theory_to_file`,
which writes the `.theory.py` to disk. **Load** reverses the trip (file → spec → repopulated
widgets) so a saved theory can be re-opened and edited.

Crucially, the UI itself does no physics. It is pure *input specification*: collect form
state → assemble a dict → emit / parse a Python source file. The only place it touches the
heavy machinery is the optional **Pre-compute** button, which builds an in-memory model from
the current form and runs a one-shot structural sanity pass (mean-field saddle + propagator
cache) so the user gets a green/red "is this theory healthy?" verdict before leaving the UI.

### Where it sits in the end-to-end pipeline

```
   ┌─────────────────────────────────────┐
   │  notebooks/theory_builder.ipynb     │   user runs one cell
   │  TheoryUI().show()                  │
   └──────────────────┬──────────────────┘
                      │  ipywidgets form (10 tabs)
                      │  user types fields/params/action/…
                      ▼
        ┌──────────────────────────────┐
        │  TheoryUI._collect()         │   snapshot form → SPEC DICT
        │  (pipeline/ui/main.py:2335)  │   {name, populations, physical_fields,
        └───────┬──────────────────────┘    parameters, functions, kernels,
                │                            cgf_terms, action_text, equations,
                │   Save                     boundary, initial, dyson, metadata, …}
                ▼
   pipeline.theory_serialize.save_theory_to_file(spec, dir)
                │
                ▼
        theories/<slug>.theory.py    ← the deliverable on disk
                │
                │   (Load reverses:  load_spec_from_file → TheoryUI.load → repopulate widgets)
                │
                ▼
   notebooks/theory_runner.ipynb  reads THEORY_NAME, imports the file,
   calls build() → model dict → pipeline.compute.compute_cumulants(...)
```

- **What feeds the UI:** the *user* (keyboard input into the form), and — on **Load** — an
  existing `theories/*.theory.py` file parsed back into a spec dict by
  `load_spec_from_file`.
- **What consumes the UI's output:** `pipeline.theory_serialize` (the serializer that turns
  the spec dict into a `.theory.py` source string). The file in turn is consumed by
  `notebooks/theory_runner.ipynb` (and the `nb_support` notebook engine), which calls
  `build()` to get the model dict and feeds it to `compute_cumulants`. The UI also writes a
  one-line scratch file `<repo>/.theories/.last_built` so the **Open in runner notebook**
  button can hand the runner the just-saved theory name.

The UI never computes a diagram. Its single output is the spec dict (and, via the
serializer, the `.theory.py` file).

---

## The math

The UI does not *implement* the MSR-JD field theory — but it is the place where the
user *declares* it, so every tab maps onto a piece of the theory. Understanding the math is
necessary to understand what each tab collects and why the validator flags what it flags.

### MSR-JD response field theory in one page

A stochastic equation of motion (EOM) for a field `φ` (a scalar `x(t)` for a temporal
theory, or a continuous field `φ(x, t)` for a spatial theory) driven by noise `η`,

```
   ∂_t φ  =  F[φ]  +  η ,        ⟨η(t) η(t')⟩ = 2D δ(t − t'),
```

is recast as a path integral by introducing a **response field** `φ̃` (Martin–Siggia–Rose /
Janssen / De Dominicis). The generating functional becomes

```
   Z = ∫ Dφ Dφ̃  exp(−S[φ, φ̃]),
```

with the MSR-JD **action**

```
   S[φ, φ̃]  =  ∫ dt [ φ̃ (∂_t φ − F[φ])  −  D φ̃² ].
```

The action is exactly the string the user types on the **Action** tab. Reading the
placeholder text from `main.py:1013`:

```
   phit * ((Dt + mu) * phi + eps * phi^3) - D * phit^2
```

This is `φ̃·(∂_t φ + μφ + ε φ³) − D φ̃²`, the action of `dφ/dt = −μφ − εφ³ + η` with
`⟨ηη⟩ = 2Dδ`. Three syntactic pieces are visible:

- `phit` — the **response field** `φ̃`. The UI never asks the user to declare it: for every
  physical field `phi` declared on the Fields tab, the framework auto-creates `phit`
  (response), `dphi` (fluctuation), and `phistar` (saddle). The UI's action validator knows
  about all three (`main.py:1684`).
- `Dt` — the **time-derivative operator** `∂_t`, composed like an algebra element:
  `(Dt + mu)*phi` means `∂_t φ + μφ`. It is one of the "builtin" identifiers the validator
  always accepts (`_ACTION_BUILTINS`, `main.py:1651`).
- `− D phit²` — the **Gaussian white noise** term. Because the action is `exp(−S)`, a
  quadratic in `φ̃` corresponds to the *second cumulant* of the noise. In the cumulant
  generating functional language (the Noise tab), this is

```
   −W[φ̃]  ⊃  −(1/n!) ∫ κ⁽ⁿ⁾ × φ̃·φ̃···φ̃   (n response legs),
```

  so a `−D φ̃²` term is `κ⁽²⁾ = 2D` (white, `δ(τ)` kernel). A `−S3 φ̃³` term would be a
  third white cumulant `κ⁽³⁾ = 3!·S3` — exactly what the Action-tab help text says
  (`main.py:1073`).

### Why each tab exists, in math terms

- **Fields** (`φ`, and auto `φ̃`, `φstar`, `dφ`). The fluctuation split `φ = φstar + δφ`
  expands the action around its mean-field saddle. The user types only `φ`; the framework
  does the split.
- **Parameters** (`μ`, `ε`, `D`, …). Coefficients appearing in `F[φ]` and in the noise.
  Their *shape* (scalar / vector indexed by one population / matrix indexed by two) follows
  the population structure.
- **Populations.** A population `A` of `size N` makes every field attached to it a length-`N`
  vector, so the action carries a `for i in A` sum. This is the `N` coupled-units case
  (`N` oscillators, `N` spins). Mathematically each population index is a label on the field;
  a matrix parameter `w[i,j]` is the coupling between unit `i` and unit `j`.
- **Functions** (`f(φ) = tanh(aφ)`, …). Non-polynomial transfer functions. The framework
  Taylor-expands `f` around the saddle of its first argument, `f(φ) = f(φstar) + f'(φstar)δφ
  + …`, so the user writes `f[i](phi[i])` and the expansion is automatic (`main.py:847`).
- **Kernels** (`K`, with `K * y = ∫ K(t−t') y(t') dt'`). Memory / convolution couplings.
  The Fourier image `K̃(ω)` is preferred because the propagator builder works in frequency
  space (`main.py:906`).
- **Noise / CGF.** Correlated, colored, or cross-correlated noise declared as cumulant
  pieces `⟨η(t)η(t')⟩ = coefficient × kernel(t−t')`. White → `dirac_delta(tau)`,
  OU-colored → `exp(-abs(tau)/tauc)`, cross-field → a coefficient like
  `rho*sqrt(D1*D2)` on two response legs `phit, psit`.
- **Mean-field equations.** The DAE residuals `LHS − RHS = 0` solved at the saddle. `Dt` on
  the LHS marks a differential equation (set `Dt → 0` at the saddle, kept symbolic for
  linear stability); no `Dt` means a purely algebraic equation. The solver finds *all* roots
  and `fixed_point_index` picks which sorted root the diagrammatic expansion uses.
- **Spatial structure** (only on the Fields tab in Spatial mode). `spatial_dim=1` makes `φ`
  a continuous field `φ(x,t)`; the action may then use the diffusion operator `Laplacian`
  multiplicatively (`D*Laplacian*phi` = `D ∇²φ`). **Derivative vertices** (KPZ `(∂_x h)²`,
  Burgers, Model B `∇²(φ²)`) put a spatial derivative *inside* a nonlinearity and need the
  **Operator-IR** mode: the action is written with `Dt()` / `Lap()` / `Dx(φ,i)` *call*
  syntax instead of bare multiplication.
- **Defaults / run metadata.** `k` (number of external legs: `k=2` is the covariance /
  power spectrum), `max_ell` (loop order: `0` tree-level / LNA, `1` one-loop, `2` two-loop),
  and the `(tau_max, tau_step)` lag grid the runner evaluates `C(τ)` on.

None of this math is *computed* by the UI — it is what the UI *collects*. The reader who
wants the actual integrals should consult the `theory-spec`, `propagator`, `mean-field`,
and `enumeration` briefs.

---

## External tools used

This subsystem touches a small number of libraries. The heavy scientific stack (SageMath,
nauty, sympy, numba, scipy, networkx) is **not** imported here — those live downstream in
the integrator. The UI's dependency surface is deliberately thin.

### `ipywidgets` — the only hard dependency of the form

```python
# pipeline/ui/main.py:27
import ipywidgets as W
# pipeline/ui/widgets.py:20
import ipywidgets as W
```

**What it is.** `ipywidgets` is the Jupyter project's library of *interactive HTML
widgets* — buttons, text boxes, dropdowns, checkboxes, tabs, sliders — that render inside a
notebook output cell and stay *live*: a Python object on the kernel side is kept in sync with
a DOM element on the browser side over the Jupyter "comm" channel. When the user types in a
box or clicks a button, the kernel-side widget's `.value` updates and any registered
callbacks fire. This is what lets a *notebook cell* host a full GUI form with no web server.

The conventional import alias here is `W` (e.g. `W.Text`, `W.Button`, `W.VBox`). The exact
widget classes this code uses:

- **Containers / layout.** `W.VBox([...])` and `W.HBox([...])` stack their children
  vertically / horizontally. `W.Tab(children=[...])` is the tab strip; `.set_title(i, str)`
  names tab `i`; `.selected_index` is the active tab (`main.py:1324`, `:1574`). `W.Layout(...)`
  is the CSS-ish layout object (`width='400px'`, `height='60px'`, `display='none'` to hide).
- **Inputs.** `W.Text` (single-line), `W.Textarea` (multi-line, used for the action and seed
  box), `W.Dropdown` (single-select; the population pickers), `W.Checkbox` (booleans:
  operator-IR, stability, dirty-guard confirms), `W.ToggleButtons` (the Temporal/Spatial
  switch), `W.IntText` / `W.FloatText` / `W.BoundedIntText` (numeric cells; *bounded* clamps
  to `[min,max]`).
- **Output / display.** `W.HTML(value=...)` renders a static HTML string (every tab's help
  text, the readiness sidebar, the cheat sheet). `W.Output(...)` is a *capture region* — the
  status panel — into which the button handlers `print(...)` (`main.py:1396`); the
  `with self._status:` context (`main.py:2642`) redirects stdout into it, and
  `.clear_output()` wipes it.
- **The event model.** Two registration methods are used everywhere:
  - `widget.observe(callback, names='value')` fires `callback` whenever `.value` changes.
    Example: `self._w_theory_mode.observe(lambda _ch: self._on_mode_change(), names='value')`
    (`main.py:409`). The lambda swallows the change-event argument (`_ch`).
  - `button.on_click(callback)` fires on click. Example:
    `self._btn_save.on_click(self._on_save)` (`main.py:1376`).
- **Styling.** `widget.add_class('tb-...')` attaches a CSS class to the widget's DOM node;
  the UI injects one inline `<style>` block (`_THEORY_BUILDER_CSS`, `main.py:63`) targeting
  those `tb-*` classes (`main.py:1421`–`1463`). `description=`, `style={'description_width':
  ...}`, and `button_style=` are widget-level cosmetics. Note the code *deliberately clears*
  `button_style` (`self._btn_save.button_style = ''`, `main.py:1455`) so the custom CSS wins
  over the built-in color presets.

The library is imported at module top level, so importing `pipeline.ui` requires ipywidgets
to be installed (it is a Jupyter-environment dependency; this brief's read environment could
not import it, which is fine — it is only needed at notebook runtime).

### `IPython.display` — pushing HTML / JS into the notebook

```python
# pipeline/ui/main.py:28
from IPython.display import display, HTML
# pipeline/ui/main.py:2240 (inside _on_open_in_runner)
from IPython.display import Javascript
```

**What it is.** IPython is the kernel powering Jupyter; `IPython.display` is its rich-output
API. `display(obj)` renders `obj` in the current cell's output (here used to render the root
`VBox` and to inject the CSS). `HTML(str)` wraps a raw HTML string so `display` renders it as
markup rather than text. `Javascript(str)` ships a snippet of JS to run in the browser — the
**Open in runner** button uses `Javascript("window.open('theory_runner.ipynb', '_blank')")`
to pop the runner notebook in a new tab (`main.py:2258`).

### `ast` — parsing the action text and `.theory.py` files

```python
# pipeline/ui/main.py:1731 (inside _validate_action)
import ast
# pipeline/ui/main.py:2579 (inside _collect_metadata)
import ast
```

**What it is.** `ast` is the Python standard-library module that parses Python *source code*
into an **Abstract Syntax Tree** — a tree of typed nodes (`ast.Name` = a bare identifier,
`ast.Call` = a function call, `ast.GeneratorExp` = a `(... for i in ...)` comprehension,
etc.) without executing it. The UI uses it for two safety-critical jobs:

1. **Action validation** (`_validate_action`, `main.py:1716`). `ast.parse(text,
   mode='eval')` turns the action string into a tree; a `SyntaxError` is caught and reported
   with its line/column (`e.lineno`, `e.offset`). A recursive `walk()` then traverses the
   tree, tracking which names are *bound* by comprehensions (`for i in A` binds `i`) and
   flagging any `ast.Name` that is neither a declared symbol nor a comprehension target as an
   *undeclared name*. This is static analysis — the action is never `eval`'d — so a typo is
   caught at type time instead of blowing up later with an opaque Sage error.
2. **Defaults-tab literal parsing** (`_collect_metadata`, `main.py:2570`). `ast.literal_eval`
   safely evaluates a *literal* dict / list / number string (no function calls, no
   arbitrary code), used to parse the seed-box `{'phi': (-3.0, 3.0)}` input.

(The companion `load_spec_from_file` in `theory_serialize.py` also uses `ast` to *read back*
a `.theory.py` by parsing its source AST rather than executing it — see Data flow below.)

### `eval` (raw built-in) — parameter-default and Defaults-dict parsing

Two spots use the raw `eval` rather than `ast.literal_eval`, but with the builtins stripped
out as a sandbox:

```python
# pipeline/ui/main.py:2342 (inside _collect, _eval_dict helper)
obj = eval(text, {'__builtins__': {}}, {})
# pipeline/ui/main.py:2427 (parameter default cell)
entry['default'] = eval(d, {'__builtins__': {}}, {})
```

Passing `{'__builtins__': {}}` as the globals removes access to `open`, `__import__`, etc.,
so a default like `[1.0, 2.0]` or `[[1,0],[0,1]]` evaluates to a Python list but
`__import__('os')` would `NameError`. (This is weaker than `ast.literal_eval` — a malicious
expression could still recurse — but the input is the user's own machine, so the threat model
is typo-resilience, not untrusted code.)

### Standard-library `os`, `re`, `subprocess`, `shutil`, `sys`

- `os` (`main.py:24`) — path joins, `makedirs`, `listdir`, `relpath`, `basename` for the
  theories directory and the save/load file plumbing.
- `re` — used *inside `_validate`* (`import re as _re`, `main.py:1996`) to word-boundary-match
  a field name in the mean-field-equation LHS text, and in `theory_serialize._slugify` (the
  filesystem-safe filename derivation the UI calls at `main.py:2193`).
- `subprocess`, `shutil`, `sys` (`widgets.py:15`–`17`) — only in `read_system_clipboard`:
  `sys.platform` to pick the OS, `shutil.which` to test for a clipboard CLI, `subprocess.run`
  to shell out to `pbpaste` / `wl-paste` / `xclip` / `xsel` / PowerShell `Get-Clipboard`.
- `pyperclip` (optional, lazily imported in `read_system_clipboard`, `widgets.py:40`) — a
  cross-platform clipboard library tried first; absence is silently tolerated and the code
  falls back to the OS CLIs.

### Downstream modules imported by the UI

These are this repository's own modules, not third-party libraries, but the UI calls into
them and the manual reader should know the boundary:

- `pipeline.theory_serialize` — `save_theory_to_file`, `render_theory_file`,
  `load_spec_from_file`, and the private `_slugify` (`main.py:38`, `:2193`). The serializer
  the UI hands its spec dict to.
- `pipeline.precompute` (`from pipeline import precompute`, `main.py:2712`) and
  `pipeline.theory_serialize.render_theory_file` — used by the **Pre-compute** button to
  `exec` an in-memory model and run the sanity pass.

---

## Components

This section is exhaustive. It covers `widgets.py` first (the primitives), then `main.py`
(the form). Signatures are quoted from the source.

### `pipeline/ui/widgets.py`

#### `read_system_clipboard() -> Optional[str]` — `widgets.py:29`

Returns the *local* system clipboard text, or `None` if no backend is available. Steps:

1. Try `import pyperclip; return pyperclip.paste()`. On any exception, fall through.
2. Pick OS-native CLI commands by `sys.platform`: `pbpaste` (macOS); `wl-paste -n`,
   `xclip -selection clipboard -o`, `xsel -b` (Linux, in that preference order); PowerShell
   `Get-Clipboard` (Windows).
3. For each candidate, skip if `shutil.which(cmd[0])` finds no binary; else
   `subprocess.run(..., capture_output=True, text=True, timeout=5)` and, on exit code 0,
   return `out.stdout.rstrip('\r\n')`.
4. If nothing works, return `None`.

**Caveat (from the docstring):** it reads the clipboard of the machine the *kernel* runs on.
Local Jupyter/VS Code → the user's machine (correct); a *remote* kernel → the server's
clipboard (wrong but harmless).

#### `paste_button(text_w, *, append=False, tooltip=..., width='34px') -> W.Button` — `widgets.py:69`

Builds the 📋 button that fills `text_w.value` from the clipboard on click. The `_on_click`
handler reads `read_system_clipboard()`; on `None` it sets the button label to `✗` and a
help tooltip (suggesting `pip install pyperclip` or a local kernel) and returns; otherwise it
sets `text_w.value = (text_w.value + text) if append else text` and restores the 📋 glyph.
**Why it exists (module comment, `widgets.py:23`):** VS Code's notebook renderer swallows
Ctrl/Cmd-V before it reaches an ipywidgets text input, so this button is the click-driven
paste path that bypasses the broken keyboard route.

#### `vector_input(label, length, defaults=None, width='70px') -> W.HBox` — `widgets.py:96`

A horizontal row of `length` `FloatText` widgets representing a vector. Returns an `HBox`
whose first child is the label; the value widgets are `box.children[1:]` and also stored on
`box._items`. The composite carries a monkey-patched `box.get_value()` returning the list of
values. **Note:** declared and exported, but *not actually used* by the current
`main.py` form (the spec uses the `default` text cell on the Parameters tab instead). Kept as
a reusable primitive.

#### `matrix_input(label, n_rows, n_cols, defaults=None, width='70px') -> W.VBox` — `widgets.py:119`

An `n_rows × n_cols` grid of `FloatText` widgets. Returns a `VBox` with `.get_value()`
returning a list-of-lists; cells also on `box._cells`. Same status as `vector_input` —
exported, not wired into the current form.

#### `expression_input(label, value='', placeholder='', validator=None, width='300px') -> W.HBox` — `widgets.py:147`

A single-line text input with a live ✓/✗ status icon. `validator(text)` must return
`(ok: bool, message: str)`; the inner `_update` callback runs it on every keystroke
(`text_w.observe(_update, names='value')`): empty input → blank icon, no validator → grey
`?`, `ok` → green `✓`, failure → red `✗` with `message` as the hover tooltip. The `HBox`
holds `[label, text, paste_button(text), indicator]` and carries `.get_value()`. **Note:**
exported but `main.py` does its action validation in the sidebar instead of per-field, so
this primitive too is currently unused by the form. It is imported in `main.py:30` but the
import is part of a bulk import; only `textarea_input`, `DynamicTable`, and `paste_button` are
actually called.

#### `textarea_input(label, value='', placeholder='', rows=8, width='600px') -> W.VBox` — `widgets.py:192`

A multi-line text area used by the **action editor** (`main.py:1008`). Returns a `VBox` of a
header (label HTML + a 📋 paste button) and the `Textarea`. Carries `.get_value()` returning
the text and exposes the raw textarea as `box._text_w` (the form reaches into `_text_w` to
attach observers and CSS classes — `main.py:1430`, `:1502`, `:2964`). Height is `rows*18px`.

#### `class DynamicTable` — `widgets.py:212`

**The workhorse.** A spreadsheet-like editor: each row is a dict of named values, columns
declared up front. Most of the form's tabs (Populations, Fields, Parameters, Functions,
Kernels, Noise, Mean-field, External-fields) are a single `DynamicTable`.

**Constructor** `__init__(self, columns, initial=None, add_label='+ add row')` —
`widgets.py:245`. `columns` is a list of column-spec dicts; the recognized keys (from the
class docstring, `widgets.py:218`):

```
{'name':             column key,
 'kind':             'text' | 'bool' | 'select' | 'int' | 'float',
 'options':          [...],              # static choices for 'select'
 'options_provider': callable() -> list, # dynamic choices, re-queried on demand
 'default':          <value>,
 'width':            '120px',
 'placeholder':      'hint text',        # text columns
 'paste':            True}               # add a 📋 button beside a text cell
```

Internal state: `self._row_widgets` (list of `{col_name: widget}` dicts — the live value
widgets), `self._row_boxes` (one `HBox` per row, for layout), `self._change_callbacks`
(observers), `self._hidden_cols` (set of column names hidden via `set_column_visible`). The
header `HBox` is built once; a `+ add row` button (`button_style='info'`) appends rows. If
`initial` is given it seeds those rows (no change-notify); else it seeds one blank row.

**`_make_widget(self, col, value=None) -> W.Widget`** — `widgets.py:272`. Factory dispatching
on `col['kind']`: `text` → `W.Text` (with placeholder); `bool` → `W.Checkbox(indent=False)`;
`select` → `W.Dropdown` whose options come from `options_provider()` if present else static
`options`, with the initial value snapped into the option list (or the first option); `int` →
`W.IntText`; `float` → `W.FloatText`. Every widget gets
`w.observe(lambda _change: self._notify_change(), names='value')` so edits propagate. An
unknown kind raises `ValueError`.

**`add_row(self, values=None, _notify=True)`** — `widgets.py:309`. Builds one widget per
column from `values`; for columns flagged `'paste': True` *and* `kind=='text'`, wraps the
input in an `HBox([w, paste_button(w)])` (the *value* widget is still the bare `w`, so
`get_rows` reads it unchanged). Appends a `✕` remove button whose click handler scans
`self._row_boxes` for the box whose last child *is* this button and pops the matching index
from both `_row_widgets` and `_row_boxes` (this identity-scan, not the captured index, is what
makes removal correct after earlier rows were already removed). Re-applies any
`_hidden_cols` display state to the new row. **Gotcha (see below):** the closure
`_make_remove(len(self._row_widgets))` captures a stale index, but the runtime identity scan
ignores it, so the captured index is effectively dead code.

**`clear(self)`** — `widgets.py:359`. Empties `_row_widgets`, `_row_boxes`, the container, and
notifies.

**`get_rows(self) -> list[dict]`** — `widgets.py:365`. Reads each row into `{name: w.value}`.
**Drops rows whose `name` cell is blank** (`if 'name' in row and not
str(row['name']).strip(): continue`) — this is how empty starter rows silently disappear from
the spec. Tables *without* a `name` column (the External-fields table, keyed `field`) are not
filtered here.

**`set_column_visible(self, name, visible)`** — `widgets.py:375`. Show/hide one column (header
cell + every row's cell, current and future) by toggling `layout.display` between `''` and
`'none'`. Used to drop the `spatial_dim` column in Temporal mode (`main.py:1533`).

**`show(self) -> W.VBox`** — `widgets.py:394`. Returns `VBox([header, rows_container,
add_btn])` — the renderable layout.

**`on_change(self, callback)`** — `widgets.py:398`. Register a callback fired on any
add/remove/cell-edit. Multiple allowed.

**`_notify_change(self)`** — `widgets.py:403`. Calls every registered callback, **swallowing
any exception** (`except Exception: pass`) so one bad observer cannot wedge the form.

**`refresh_dropdown_options(self, col_name)`** — `widgets.py:410`. Re-queries a `select`
column's `options_provider()` and pushes the new option list onto every existing row's
dropdown, *preserving the prior selection when it is still valid* else falling back to the
first option. Wraps the assignment in try/except because some ipywidgets versions reject
setting `.options` while the current `.value` is momentarily invalid; the fallback assigns
`value` first then `options`. This is how a new population declared on the Populations tab
instantly appears in the index dropdowns on every other tab.

### `pipeline/ui/main.py`

#### Module-level

- `_DEFAULT_THEORIES_DIR` (`main.py:47`) — `<cwd>/../theories`, absolute. Because notebooks
  run from `notebooks/`, this resolves to the repo's `theories/` directory.
- `_THEORY_BUILDER_CSS` (`main.py:63`) — one big inline `<style>` string in the `tb-*`
  namespace (palette, spacing scale, button palette, tab strip, validation sidebar, status
  panel, table striping). Injected once via the `_CSS_INJECTED` module guard (`main.py:244`).
- `_normalize_cgf_rows(rows) -> list[dict]` (`main.py:249`). Converts raw UI **Noise/CGF**
  rows into spec-form rows. The UI carries `response_legs` as a comma-separated string; this
  splits it: empty → untouched; single leg → emit legacy `response_field=<leg>` (and
  `response_legs=None`); multi-leg → keep `response_legs=[...]` *and* stamp `response_field`
  with the first leg's name as a cosmetic back-compat fallback for older loaders.

#### `class TheoryUI` — `main.py:289`

The whole form. Public surface: `__init__`, `show`, `spec`, `load`, `last_saved` (attr),
`theories_dir` (attr). Everything else is private.

**`__init__(self, theories_dir=None)`** — `main.py:296`. Resolves `theories_dir` (default
`_DEFAULT_THEORIES_DIR`), `makedirs(exist_ok=True)`, initializes `last_saved=None`,
`_dirty=False` (unsaved-changes flag), and `_loaded_extras` — a dict of `{field_latex,
function_latex, kernel_latex}` carry-through maps keyed by name. The latter exists because the
`latex` column was *removed* from the UI (2026-05-27) but old theory files may still carry
custom latex strings; `load()` stashes them here and `_collect()` re-injects them so a
load→edit→save round-trip does not lose them. Then calls `_build_widgets()`.

**Population helpers.**

- `_pop_names(self) -> list[str]` (`main.py:321`) — current non-blank population names from
  the Populations table. Used as the `options_provider` for population dropdowns elsewhere.
- `_pop_size_map(self) -> dict[str,int]` (`main.py:330`) — `{name: size}` for populations with
  a positive integer size.
- `_autofill_default_templates(self)` (`main.py:345`) — walks every Parameters row; if its
  `default` cell is *empty* and its `index_1`/`index_2` dropdowns name populations of known
  size, fills a placeholder template (`'[, , ]'` for a length-3 vector, `'[[, , ],[, , ]]'`
  for a 2×3 matrix). Preserves any non-empty user value. Reaches directly into
  `_tbl_parameters._row_widgets` (not `get_rows`) because it needs to *write* the widget
  `.value`.

**`_build_widgets(self)`** — `main.py:381`. The 700-line constructor of the entire form.
Builds, in order:

- **Tab 1 Model** — `_w_name` (`Text`), `_w_description` (`Textarea`), and `_w_theory_mode`
  (`ToggleButtons` of `['Temporal (ODE)', 'Spatial (PDE)']`). The mode toggle's observer
  calls `_on_mode_change`.
- **Tab 2 Populations** — `_tbl_populations` (`DynamicTable`: name / size / description; no
  starter rows — a scalar theory needs no populations).
- **Tab 3 Fields** — `_tbl_physical` (`DynamicTable`: name / population(select via
  `_pop_names`) / spatial_dim(int) / description; one starter row `phi`). Plus the
  **spatial-structure panel**: `_w_spatial_dim` (`BoundedIntText` 0–3), `_w_spatial_apply`
  (`Button` → `_apply_spatial_dim_to_all`, which bulk-writes every row's `spatial_dim` and
  re-runs validation), `_w_boundary_mode` (`Dropdown` infinite/periodic), `_w_boundary_length`
  (`Text`), `_w_initial_mode` (`Dropdown` stationary), `_w_operator_ir` (`Checkbox`),
  `_w_dyson_order` (`BoundedIntText` 0–99), `_w_reference_diffusion` (`Text`). The spatial
  controls are then *grouped* into `_w_spatial_block` (`main.py:700`) so the Temporal/Spatial
  toggle can hide them all at once.
- **Tab 4 Parameters** — `_tbl_parameters` (name / index_1 / index_2 / domain / default
  (paste-enabled) / description; one starter row `mu`). `index_1`/`index_2` are dropdowns over
  `['—'] + populations`. Default cells auto-template when an index changes (wired at
  `main.py:1343`).
- **Tab 5 Functions** — `_tbl_functions` (name / population / args / expression(paste) /
  description; no starter rows).
- **Tab 6 Kernels** — `_tbl_kernels` (name / index_1 / index_2 / time_expr(paste) /
  freq_image(paste); no starter rows).
- **Tab 7 Noise** — `_tbl_cgfs` (name / response_legs / order / coefficient(paste) /
  kernel(paste); no starter rows).
- **Tab 8 Action** — `_w_action = textarea_input('Action S', ...)` (`main.py:1008`).
- **Tab 9 Mean-field** — `_tbl_mfeqs` (lhs(paste) / rhs(paste) / population; no starter
  rows), `_w_fpi_default` (`BoundedIntText` — default `fixed_point_index`), `_w_stability_on`
  (`Checkbox`), `_w_seed_box` (`Textarea` — multi-start Newton seed dict).
- **Tab 10 Defaults** — `_w_k_default`, `_w_ell_default` (`BoundedIntText`), `_w_tau_max`,
  `_w_tau_step` (`FloatText`), and `_tbl_ext_fields` (`DynamicTable`: field / leaf_index; two
  starter rows `phi[1], phi[2]`).

Then composes the `W.Tab` (`main.py:1324`) from `_all_tabs` (a list of `(widget, title)`
pairs — `_apply_visible_tabs` picks the visible subset by theory type), wires the live
plumbing (`_on_populations_changed` refreshes all dependent dropdowns; the parameter table
re-templates defaults on index change), builds the bottom bar, attaches the readiness sidebar
and CSS classes, registers `_mark_changed` on every table and standalone widget, and finally
calls `_refresh_save_hint()` and `_on_mode_change(mark_dirty=False)` for the initial render.

**Theory-type (Temporal / Spatial) machinery.**

- `_is_spatial_mode(self) -> bool` (`main.py:1522`) — `True` iff the toggle value starts with
  `'Spatial'`.
- `_on_mode_change(self, mark_dirty=True)` (`main.py:1525`) — shows/hides every spatial
  control. Temporal: hide `_w_spatial_block`, hide the `spatial_dim` column, force every
  field's `spatial_dim=0`, and *clear* the operator-IR / Dyson / boundary / initial widgets so
  a saved time-only theory carries no spatial structure. Spatial: show the block, show the
  column, and *clear the Kernels and Noise tables* (spatial theories take only Gaussian white
  independent noise and have no convolution kernels, so neither can leak into a saved PDE
  theory), bump `_w_spatial_dim` to ≥1. Then `_apply_visible_tabs()` and `_refresh_validation`.
- `_apply_visible_tabs(self)` (`main.py:1560`) — rebuilds `self._tabs.children` to the visible
  subset (Spatial hides the Kernels + Noise tabs), renumbers titles `1.`, `2.`, …, and
  preserves the current selection when it stays visible.

**Bottom bar.**

- `_build_bottom_bar(self) -> W.VBox` (`main.py:1582`) — composes the save-path field +
  derived-filename hint, the four primary action buttons (`_btn_save`, `_btn_preview`,
  `_btn_precompute`, `_btn_open_runner`), and the destructive-action gate: two confirm
  checkboxes (`_chk_confirm_reset`, `_chk_confirm_load`) plus the Reset and Load rows. Creates
  `_btn_open_runner` here and wires it to `_on_open_in_runner`.

**Action validation.**

- `_ACTION_BUILTINS` (`main.py:1651`) — a `frozenset` of identifiers that may appear in the
  action without being declared: operators (`Dt`, `Conv`), math (`exp`, `log`, trig, `sqrt`,
  `abs`, `pi`, `e`, `I`, `oo`), `heaviside`/`dirac_delta`/`sign`, indexing helpers
  (`sum`, `range`, `len`). Keeping it explicit catches typos like `hevyside` rather than
  silently treating them as free variables.
- `_action_known_names(self, spec) -> (set, set)` (`main.py:1663`) — builds
  `(known_global_names, known_populations)`. Known globals = `_ACTION_BUILTINS` + every
  declared parameter + every physical field with its auto companions (`<name>`, `<name>t`,
  `d<name>`, `<name>star`) + kernels + functions + each population name and `pop_<name>` +
  the legacy alias `pop`. For spatial theories it adds `Laplacian`; if operator-IR is on it
  adds `Lap`, `Dx`, `Gradient`, `GradX`.
- `_validate_action(self, action_text, spec) -> list[(severity, message)]` (`main.py:1716`) —
  the AST walk described under [External tools](#external-tools-used). Catches syntax errors
  (with line/col), undeclared names (first 5 shown, `+N more`), `for i in X` where `X` is not
  a declared population (`bad_pops`), and the foot-gun of iterating over `pop` when no
  populations are declared. The recursive `walk(node, bound)` tracks comprehension targets and
  lambda args so inner indices are not falsely reported.

**Cross-tab validation + cheat sheet.**

- `_validate(self) -> dict` (`main.py:1830`) — walks the *collected* spec and returns
  `{tab_index_1based: [(severity, message), …]}` with severity in `{warn, error, info}`. The
  docstring is explicit about what it does check (undeclared-population references on
  fields/params/kernels — the "silent `n_populations=0` bug class"; the action AST
  cross-reference; the Defaults dict parse failure; reserved-name collisions; per-field
  mean-field-equation coverage for coupled theories) and what it deliberately does *not* nag
  about (theory-name-still-default, stability-off, action-blank). Reserved names: `t`,
  `omega`, `Dt`, `delta_D`, `delta_Dp` everywhere; plus `k`, `Laplacian`, `x`, `y`, `z` in
  spatial theories. Spatial scope notes are emitted as `info`/`warn` (k≥3 needs
  `spatial_points=`; loop order > 2 is expensive). It wraps `_collect()` in try/except so a
  malformed Defaults dict becomes a validation finding instead of a crash.
- `_build_cheat_sheet(self) -> str` (`main.py:2010`) — an HTML reference of all declared
  names by tab. Fields render as *scalars* (`n`, `nt`, `nstar`) when they have no valid
  population, or *indexed* (`n[i] for i in pop_X`, response `nt[i]`, saddle `nstar[i]`) when
  they do. Filters out blank starter rows.
- `_refresh_validation(self)` (`main.py:2097`) — re-renders the sidebar: a per-tab readiness
  badge strip (red `●` if any error, amber if any warn, green otherwise — note `info` does
  *not* trip a badge), a detailed `<ul>` of findings, and the cheat sheet. The "No problems
  detected" green line only appears once the user has typed at least one field / parameter /
  action, so a blank starter form does not lie green.
- `_refresh_save_hint(self)` (`main.py:2176`) — updates the `→ writes theories/<file>` hint
  next to the save-path field, mirroring `_on_save`'s path logic and using
  `theory_serialize._slugify` for the auto-name.

**Dirty / unsaved-changes tracking.**

- `_mark_changed(self)` (`main.py:2204`) — sets `_dirty=True` and refreshes the validation
  panel. Registered on every table `on_change` and every standalone widget `observe`.
- `_check_dirty_guard(self, which) -> bool` (`main.py:2217`) — returns `True` (action allowed)
  if the form is clean, or if the matching confirm checkbox (`_chk_confirm_reset` /
  `_chk_confirm_load`) is ticked. This gates Reset and Load so unsaved work is not silently
  destroyed.

**Bridges and error formatting.**

- `_on_open_in_runner(self, _btn)` (`main.py:2231`) — the **Open in runner** bridge. If
  nothing has been saved, prints a hint. Otherwise derives the slug from `self.last_saved`,
  writes it (one line) to `<theories_dir>/../.theories/.last_built`, and `display`s a
  `Javascript("window.open('theory_runner.ipynb', '_blank')")`. The runner reads
  `.theories/.last_built` when its `THEORY_NAME` is blank. Degrades gracefully (printed
  instruction) when JS is unavailable (e.g. nbconvert export).
- `_humanize_error(self, exc, context) -> str` (`main.py:2267`) — turns an exception into one
  human line, tailored by `context ∈ {save, preview, precompute}`. Special-cases
  `SyntaxError` (line/col), `NameError` ("Declare it on the Parameters/Fields/Kernels tab
  first"), `KeyError`, `ValueError`; falls back to `class: message`.

**Public API.**

- `show(self)` (`main.py:2311`) — injects the CSS once (`_CSS_INJECTED` guard), refreshes the
  validation sidebar, and `display`s the root `VBox`.
- `spec(self) -> dict` (`main.py:2330`) — returns `self._collect()` (the serializer-ready spec
  dict).

**Form-state collection (the core data-out path).**

- `_collect(self) -> dict` (`main.py:2335`) — snapshots every tab into the spec dict. Walks
  each table via `get_rows()`, drops blank rows, and maps the UI shapes onto the serializer's
  shapes:
  - `_index_list(row)` (nested, `main.py:2352`) translates the `(index_1, index_2)` dropdowns
    (`'—'` = none) into the `indexed_by` list (`[]` scalar, `[pop]` vector, `[pop1, pop2]`
    matrix).
  - Populations → `{name, size(≥1), description?}`.
  - Physical fields → `{name, population?, latex?(from _loaded_extras), spatial_dim?(only if
    >0), description?}`.
  - Parameters → `{name, indexed_by?, domain?, default?(eval'd literal or raw string),
    description?}`.
  - Functions → `{name, args(split on comma), expression, population?, latex?, description?}`.
  - Kernels → `{name, indexed_by?, time_expr?, freq_image?, latex_name?}`.
  - CGF/Noise → `_normalize_cgf_rows(...)` output.
  - Spatial blocks (`boundary`, `initial`, `operator_ir`, `dyson`, `reference_diffusion`)
    emitted **only when at least one field is spatial**, matching the serializer's
    "emit-only-when-present" rule.
  - Mean-field → `equations` list of `{lhs, rhs, population(None when '—')}`, only for rows
    with both lhs and rhs non-blank.
  - Plus `name`, `n_populations` (derived `len`), `description`, `response_fields=[]`
    (auto-generated downstream), `stability_analysis`, `default_fundamental={}` (deliberately
    empty — per-parameter `default=` now carries values), and `metadata`.
- `_collect_metadata(self) -> dict` (`main.py:2570`) — builds the METADATA dict from the
  Defaults-tab widgets: `k_default`, `ell_default`, `tau_max`, `tau_step`,
  `recommended_external_fields` (list of `(field, leaf_index)` tuples),
  `fixed_point_index_default` (always written, even when 0, so reload is unambiguous), and an
  optional `seed_box_default` (parsed by `_safe_dict` via `ast.literal_eval`; list values
  coerced to tuples; bad input falls through as the raw string so Preview surfaces the
  problem).

**Button handlers.**

- `_on_preview(self, _btn)` (`main.py:2640`) — `_collect()` → `render_theory_file(spec)` →
  print the source into the status panel (or a humanized error).
- `_on_save(self, _btn)` (`main.py:2650`) — `_collect()`; resolve the target path (explicit
  field, or `theories_dir`, joining a relative non-`.py` base); `save_theory_to_file(spec,
  target)`; set `last_saved`, clear `_dirty` and both confirm checkboxes; print the path and
  the `THEORY_NAME = <slug>` instruction; refresh validation.
- `_on_reset(self, _btn)` (`main.py:2689`) — gated by `_check_dirty_guard('reset')`; on
  confirm, re-runs `_build_widgets()` to wipe state, clears `_dirty`, re-`show()`s.
- `_on_precompute(self, _btn)` (`main.py:2704`) — `_collect()` → `render_theory_file(spec)` →
  `exec(compile(src, '<precompute-spec>', 'exec'), ns)` → `ns['build']()` to get an in-memory
  model **without writing a file** → `pipeline.precompute(model, verbose=True)`. Prints a
  summary: MF check (✓ PASS / ✗), saddle values, propagator (cached / FAILED), cache dir,
  wall time. (The result dict keys come from `_precompute.precompute`, `_precompute.py:78`.)

**Loading existing theories.**

- `_list_theory_files(self) -> list[str]` (`main.py:2758`) — sorted `*.theory.py` filenames in
  `theories_dir` (for the Load dropdown).
- `_on_load(self, _btn)` (`main.py:2769`) — gated by `_check_dirty_guard('load')`; refreshes
  the dropdown so freshly-saved files appear; `load_spec_from_file(path)` →
  `self.load(spec)`; clears `_dirty` and confirm checkboxes; refreshes validation + save hint.
- `load(self, spec_or_path)` (`main.py:2813`) — the big *repopulate-the-form-from-a-spec*
  routine. Accepts a spec dict or a path. Clears each table and re-adds rows from the spec,
  refreshing all population dropdowns. Handles the historical wrinkles:
  - Strips the auto-prefix: a field stored as `dn` with `natural_name='n'` displays as `n`
    (`display_name = f.get('natural_name') or f.get('name')`).
  - Stashes any `latex` strings into `_loaded_extras` for round-trip preservation.
  - **Auto-skips saddle parameters** (`mean_field=True` or name matching `<natural>star`) —
    the framework re-creates them from the field declarations.
  - Maps `indexed_by` back to `(index_1, index_2)` dropdowns; `default` back to its text
    (`repr(d)` for non-strings).
  - CGF `response_legs` (list/str) round-trips to the comma-string cell, falling back to
    legacy `response_field`.
  - Mean-field: prefers the new `equations` list; converts legacy `mf_equations`
    (`{saddle, rhs}`) to `{lhs='<natural>[i]', rhs, population}` by stripping the trailing
    `star` and looking up the field's population.
  - Restores `stability_analysis`, the spatial `boundary`/`initial`/`operator_ir`/`dyson`/
    `reference_diffusion` widgets (absent keys reset to time-only defaults), then picks the
    Temporal/Spatial mode from the loaded fields and applies it (which fires or re-applies
    `_on_mode_change`), then drives every Defaults-tab widget from `metadata`.

---

## Data structures

### The column-spec dict (input to `DynamicTable`)

```python
{'name': 'index_1', 'kind': 'select', 'options_provider': _pop_opts_with_none,
 'default': '—', 'width': '100px'}            # main.py:719
{'name': 'default', 'kind': 'text', 'placeholder': '1.0  /  [.., ..]  /  [[..]]',
 'width': '300px', 'paste': True}             # main.py:728
```

Fields: `name` (the dict key produced by `get_rows`), `kind` (widget type), `options` /
`options_provider` (for `select`), `default`, `width`, `placeholder` (text), `paste` (add a
📋 button). See `DynamicTable` docstring, `widgets.py:218`.

### The row dict (output of `DynamicTable.get_rows()`)

A flat `{col_name: value}` dict per row, e.g. a Parameters row:
`{'name': 'mu', 'index_1': '—', 'index_2': '—', 'domain': 'real', 'default': '1.0',
'description': ''}`. Blank-`name` rows are dropped.

### `_loaded_extras` (latex carry-through)

```python
self._loaded_extras = {'field_latex': {}, 'function_latex': {}, 'kernel_latex': {}}
# main.py:313 — keyed by name → custom latex string
```

A per-kind `{name: latex}` map so latex strings on loaded specs survive a load→save trip even
though the latex column was dropped from the UI.

### The spec dict (output of `_collect()`, input to the serializer)

The central data structure. Built at `main.py:2498`. Keys and shapes:

```python
{
  'name':            str,                 # _w_name
  'populations':     [{'name','size','description'?}, ...],
  'n_populations':   int,                 # derived len(populations)
  'description':     str,
  'response_fields': [],                  # always empty — auto-generated downstream
  'physical_fields': [{'name','population'?,'spatial_dim'?,'latex'?,'description'?}, ...],
  'parameters':      [{'name','indexed_by'?,'domain'?,'default'?,'description'?}, ...],
  'functions':       [{'name','args':[...],'expression','population'?,'latex'?,'description'?}, ...],
  'kernels':         [{'name','indexed_by'?,'time_expr'?,'freq_image'?,'latex_name'?}, ...],
  'cgf_terms':       [{'name','response_field','response_legs'?,'order','coefficient','kernel'}, ...],
  'action_text':     str,                 # the action textarea
  'equations':       [{'lhs','rhs','population'(or None)}, ...],
  'stability_analysis': bool,
  'default_fundamental': {},              # deliberately empty
  'metadata':        {...},               # see below
  # spatial-only, present only when a field is spatial:
  'boundary':            {'mode': 'infinite'|'periodic', 'length'?: float|str},
  'initial':             {'mode': 'stationary'},
  'operator_ir':         True,            # only when ticked
  'dyson':               {'mode':'fixed','order':N},   # only when order>0
  'reference_diffusion': float,           # only when set
}
```

The `metadata` sub-dict (from `_collect_metadata`, `main.py:2598`):

```python
{
  'k_default': int, 'ell_default': int, 'tau_max': float, 'tau_step': float,
  'recommended_external_fields': [(field, leaf_index), ...],   # only if non-empty
  'fixed_point_index_default': int,                             # always written
  'seed_box_default': {var: (low, high), ...},                 # only if provided
}
```

### The precompute result dict (consumed by `_on_precompute`)

From `_precompute.precompute` (`_precompute.py:78`): `{'mf_check', 'sanity_ok', 'mf_values',
'taylor_order', 'cache_dir', 'propagator_built', 'wall_seconds', 'log'}`. The UI reads
`mf_check` (`'PASS'` → ✓), `mf_values`, `propagator_built`, `cache_dir`, `wall_seconds`.

---

## Data flow

### Forward (authoring → file)

1. User edits widgets. Every edit fires `_mark_changed` → `_dirty=True` + sidebar refresh.
2. **Save**: `_on_save` → `_collect()` builds the **spec dict** (above) →
   `theory_serialize.save_theory_to_file(spec, target)` → `render_theory_file(spec)` emits the
   `.theory.py` source → written to disk → `last_saved` set, scratch hint printed.

Concrete example. With the form filled like the placeholder (one field `x` in population
`pop`; params `mu`, `eps`, `D`; the OU-quartic action), `_collect()` yields a spec whose
serialized form is exactly the on-disk file shown earlier
(`theories/ou_quartic.theory.py`):

```python
def build():
    return (
        TemporalTheoryBuilder('OU Quartic (white noise)')
        .population('pop', size=1)
        .physical_field('x', population='pop', description='variable')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('eps', default=0.02, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
        .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
        .build()
    )
DEFAULT_FUNDAMENTAL = {'mu': 1.0, 'eps': 0.02, 'D': 1.0}
METADATA = {'k_default': 2, 'ell_default': 0,
            'recommended_external_fields': [('dx', 1), ('dx', 1)],
            'tau_max': 8.0, 'tau_step': 0.5, 'fixed_point_index_default': 0}
```

(The serializer chooses `SpatialTheoryBuilder` vs `TemporalTheoryBuilder` by whether any field
carries `spatial_dim ≥ 1`; see `theory_serialize.render_theory_file`, `theory_serialize.py:370`.
Saddle parameters `xstar` are *not* emitted — `build()` auto-creates them.)

### Reverse (file → form)

3. **Load**: `_on_load` → `load_spec_from_file(path)` (parses the `.theory.py` *source AST*,
   reconstructing the spec dict without executing it — `theory_serialize.py:528`) →
   `self.load(spec)` clears and repopulates every table/widget. Round-trip fidelity: action
   text, equation RHS strings, function expressions, and kernel exprs are stored as string
   literals so they survive verbatim; comments and formatting are lost (AST round-trip, not
   textual). `mf_equations` (legacy) auto-converts to `equations`.

### Pre-compute (sanity pass, no file)

4. **Pre-compute**: `_on_precompute` → `_collect()` → `render_theory_file(spec)` →
   `exec` the source in a fresh namespace → `ns['build']()` → in-memory model dict →
   `precompute(model, verbose=True)` → printed PASS/FAIL summary. No `.theory.py` is written.

### Open-in-runner bridge

5. **Open in runner**: `_on_open_in_runner` writes the saved slug to `.theories/.last_built`
   and JS-opens `theory_runner.ipynb`, which reads that scratch file when `THEORY_NAME` is
   blank.

---

## Gotchas & caveats

- **macOS / fork safety is irrelevant here.** The UI runs purely on the kernel main thread —
  no multiprocessing, no fork. (The fork crash documented in project memory lives in the
  spatial/temporal *integrator* paths, not the UI.) Pre-compute runs `precompute` in-process.

- **`eval` with stripped builtins, not `ast.literal_eval`, for parameter defaults**
  (`main.py:2342`, `:2427`). `eval(text, {'__builtins__': {}}, {})` is a *weaker* sandbox than
  `literal_eval` — it blocks name lookups (`__import__` → NameError) but still evaluates
  arbitrary *operators*. The threat model is the user's own typos on the user's own machine,
  so this is acceptable, but a reader should not mistake it for safe-evaluation of untrusted
  input. The Defaults seed-box, by contrast, *does* use `ast.literal_eval` (`main.py:2586`).

- **Blank-row dropping is `name`-keyed.** `DynamicTable.get_rows` only drops a row when it has
  a `name` column that is blank (`widgets.py:370`). The External-fields table is keyed `field`
  (no `name` column), so its blank rows are *not* auto-dropped there — `_collect_metadata`
  re-filters on `field` instead (`main.py:2607`). A future table without a `name` column would
  silently keep blank rows unless it filters itself.

- **The remove-button captured index is dead code.** `add_row` captures
  `_make_remove(len(self._row_widgets))` (`widgets.py:341`) but the handler ignores `_idx` and
  finds the row by *identity scan* of `_row_boxes` (`widgets.py:332`). The capture is harmless
  but misleading — removal correctness comes from the identity scan, not the index.

- **`_notify_change` swallows all exceptions** (`widgets.py:407`). A buggy `on_change`
  callback fails silently. This protects the form from wedging but can hide validator bugs.

- **`refresh_dropdown_options` ordering hazard.** Some ipywidgets versions raise when you set
  `.options` while the current `.value` is not in the new options; the code wraps this in
  try/except and retries with value-first-then-options (`widgets.py:434`). If you add a new
  population-aware dropdown, route its refresh through this method, not a raw `.options =`.

- **Switching Temporal → Spatial silently clears the Kernels and Noise tables**
  (`main.py:1548`). Intentional (spatial supports only Gaussian white independent noise), but
  a user who typed kernels/noise and then flips to Spatial loses them with no undo. Likewise
  Spatial → Temporal force-zeros every `spatial_dim` and clears operator-IR / Dyson / boundary
  (`main.py:1534`).

- **`latex` columns were removed from the UI (2026-05-27)** but are preserved on round-trip
  via `_loaded_extras` (`main.py:313`, re-injected in `_collect`). If a user *renames* a field
  whose loaded latex was keyed by the old name, the latex is dropped (the carry-through is
  keyed by name).

- **`default_fundamental` is always emitted empty** by `_collect` (`main.py:2544`). The
  per-parameter `default=` declarations now carry the suggested values; the empty
  `DEFAULT_FUNDAMENTAL` in the file is a legacy-shape placeholder. A loaded file's
  `DEFAULT_FUNDAMENTAL` is *ignored* on load (`main.py:3001` comment).

- **`fixed_point_index_default` is always written, even when 0** (`main.py:2620`) — so reload
  is unambiguous. Most other optional keys are emitted only when non-default.

- **Operator-IR is load-bearing and silent.** Per the serializer comment
  (`theory_serialize.py:455`), without `.operator_ir()` a re-saved KPZ/Burgers/Model B theory
  *silently loses* the `Dt()`/`Lap()`/`Dx()` lowering and computes the wrong physics. The UI
  emits it only when the checkbox is ticked, so a user who forgets to tick it for a derivative
  vertex gets a wrong-but-syntactically-valid theory. The validator does *not* detect this
  (it only allowlists the call names when the box is already on).

- **The validator can be fooled by free-form mean-field LHS.** The "every field needs an
  equation" check (`main.py:1991`) is a substring/word-boundary match on the LHS text, not a
  parse — a field whose name appears only inside a function call or comment-like string could
  pass spuriously, and it only fires for ≥2 fields (coupled theories).

- **CSS injected once per kernel session** via `_CSS_INJECTED` (`main.py:244`). Multiple
  `TheoryUI()` instances in one kernel share the stylesheet; if you edit `_THEORY_BUILDER_CSS`
  you must restart the kernel to see it.

- **`expression_input`, `vector_input`, `matrix_input` are exported but unused** by the
  current form. They are reusable primitives; do not assume editing them affects the live UI.

- **Clipboard reads the *kernel's* machine** (`widgets.py:36`). Remote-kernel users get the
  server's clipboard (or `None` and a `✗` glyph). The 📋 buttons exist specifically because VS
  Code's notebook renderer swallows Ctrl/Cmd-V (`widgets.py:23`).

- **Open-in-runner depends on JS + relative paths.** `window.open('theory_runner.ipynb', ...)`
  assumes the runner sits next to the builder in the Jupyter file tree and that JS execution
  is allowed; it degrades to a printed instruction otherwise (`main.py:2262`).

---

## Glossary

- **ipywidgets** — Jupyter's interactive-widget library. Python objects on the kernel side
  synced to live HTML controls in the notebook. Imported as `W`.
- **IPython.display** — IPython's rich-output API; `display`, `HTML`, `Javascript`.
- **widget `.observe(cb, names='value')`** — register a callback fired when a widget's value
  changes. **`.on_click(cb)`** — the button-click equivalent.
- **`W.Output` / `with self._status:`** — a capture region in the notebook into which
  `print` is redirected; the UI's status/log panel.
- **AST (Abstract Syntax Tree)** — the tree of Python syntax nodes produced by `ast.parse`,
  walked to validate the action without executing it.
- **`ast.literal_eval`** — safe evaluation of a *literal* (dict/list/number/string) with no
  code execution. Used for the seed box.
- **spec dict** — the flat Python dict `_collect()` produces from the form; the serializer's
  input and `load`'s output.
- **`.theory.py` file** — the on-disk deliverable: a Python module with a `build()` function
  that constructs the theory and `DEFAULT_FUNDAMENTAL` / `METADATA` module globals.
- **MSR-JD action `S`** — Martin–Siggia–Rose–Janssen–De Dominicis action; the response-field
  path-integral weight `exp(−S)`. The Action tab collects `S` as text.
- **response field `phit` (`φ̃`)** — the MSR auxiliary field conjugate to a physical field;
  auto-created per declared field. Also `dphi` (fluctuation), `phistar` (saddle).
- **saddle / mean-field value `phistar`** — the steady-state value the action is expanded
  around; solved from the Mean-field tab's DAE.
- **DAE (Differential-Algebraic Equation)** — the `LHS − RHS = 0` mean-field residual form;
  `Dt` on the LHS marks a differential equation, its absence an algebraic one.
- **CGF (Cumulant Generating Functional)** — the internal name of the Noise tab; rows declare
  noise cumulant pieces `⟨ηη⟩ = coefficient × kernel(τ)`.
- **population** — a named group of `size N` identical units; makes attached fields length-`N`
  vectors and introduces `for i in pop` sums in the action.
- **kernel (memory kernel)** — a convolution coupling `K * y = ∫ K(t−t')y(t')dt'`, declared
  by its time form or (preferred) its Fourier image `K̃(ω)`.
- **`Laplacian` / Operator-IR** — in spatial theories, `Laplacian` is the diffusion operator
  used multiplicatively (`D*Laplacian*phi` = `D∇²φ`). Operator-IR mode switches to
  `Dt()`/`Lap()`/`Dx()` *call* syntax, required for derivative vertices (KPZ/Burgers/Model B).
- **Dyson order / reference diffusion** — for coupled theories with *unequal* diffusion
  constants, `dyson_order=N` dresses the propagator to `O(𝒟̂ᴺ)` in the off-diagonal diffusion;
  `reference_diffusion` is the scalar `D0` in the `𝒟 = D0·I + 𝒟̂` split.
- **`k` / `max_ell`** — run-metadata knobs: `k` external legs (`k=2` = covariance / power
  spectrum), `max_ell` loop order (`0` tree, `1` one-loop, `2` two-loop).
- **`fixed_point_index`** — which sorted mean-field root the diagrammatic expansion uses when
  the DAE has several.
- **precompute** — `pipeline.precompute(model)`: the structural sanity pass (expand at order 2,
  verify saddle, build+cache propagator) the **Pre-compute** button runs.
- **slugify** — `theory_serialize._slugify`: filesystem-safe lowercase name derivation for the
  `<slug>.theory.py` filename.
- **dirty flag / dirty guard** — `_dirty` tracks unsaved edits; Reset/Load are gated behind a
  confirm checkbox while dirty.

---

## Proposed manual subsections

1. **What the Theory Builder UI is** — the point-and-click front door; where it sits between
   the user and `theory_runner.ipynb`.
2. **Launching it** — `from pipeline.ui import TheoryUI; TheoryUI().show()` in
   `notebooks/theory_builder.ipynb`; what `show()` does.
3. **ipywidgets in 5 minutes** — the kernel↔browser widget model, `observe`/`on_click`,
   `VBox`/`HBox`/`Tab`, `Output` status panels, `add_class` styling. (For the physics reader
   who has never used ipywidgets.)
4. **The tab-by-tab walkthrough** — Model, Populations, Fields (+ spatial structure),
   Parameters, Functions, Kernels, Noise, Action, Mean-field, Defaults — what each collects
   and the underlying MSR-JD piece.
5. **Temporal vs Spatial mode** — the toggle, what it shows/hides/clears, the reserved-name
   rules.
6. **The `DynamicTable` primitive** — columns, kinds, dynamic dropdowns, add/remove,
   `get_rows`, hidden columns, the population-aware `options_provider` wiring.
7. **The readiness sidebar** — per-tab badges, the action AST validator, the
   "undeclared-population" / reserved-name checks, the declared-names cheat sheet.
8. **From form to file: the spec dict and `theory_serialize`** — `_collect()`, the spec dict
   shape, `render_theory_file`, the emitted `.theory.py`, the round-trip via
   `load_spec_from_file`.
9. **Save / Preview / Load / Reset / Pre-compute / Open-in-runner** — the bottom bar, the
   dirty-guard, the in-memory precompute sanity pass, the runner bridge.
10. **Gotchas** — operator-IR is load-bearing-and-silent, mode-switch data loss, the weak
    `eval` sandbox, latex carry-through, clipboard caveats.

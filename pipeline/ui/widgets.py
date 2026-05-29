"""
pipeline.ui.widgets — small, reusable ipywidgets primitives.

These are the building blocks the tab editors compose:

- :func:`vector_input` — row of ``FloatText`` widgets for a length-N vector
- :func:`matrix_input` — N×M grid of ``FloatText`` widgets
- :func:`expression_input` — ``Text`` widget with a green/red ✓/✗ indicator
- :class:`DynamicTable` — table editor with ``add row`` / ``remove`` buttons,
                          one column per declared field
- :func:`textarea_input` — multi-line text input (action editor uses this)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

import ipywidgets as W


# ── Clipboard paste (works around VS Code's notebook renderer, which
#    swallows Ctrl/Cmd-V before it reaches an ipywidgets text input).
#    Button *clicks* still work in VS Code, and when the kernel runs on
#    the same machine as the editor (the normal case) Python can read
#    the real system clipboard — so a 📋 button reads it and fills the
#    box directly, bypassing the broken keyboard path. ────────────────
def read_system_clipboard() -> Optional[str]:
    """Return the LOCAL system clipboard text, or ``None`` if no backend
    is available.  Tries ``pyperclip`` first, then OS-native CLIs
    (``pbpaste`` on macOS; ``wl-paste`` / ``xclip`` / ``xsel`` on Linux;
    PowerShell ``Get-Clipboard`` on Windows).

    Caveat: reads the clipboard of the machine the *kernel* runs on.
    For a local VS Code / Jupyter session that's your machine (correct);
    for a remote kernel it would be the server's clipboard.
    """
    try:                       # pyperclip is cross-platform if installed
        import pyperclip
        return pyperclip.paste()
    except Exception:
        pass

    if sys.platform == 'darwin':
        cmds = [['pbpaste']]
    elif sys.platform.startswith('linux'):
        cmds = [['wl-paste', '-n'],
                ['xclip', '-selection', 'clipboard', '-o'],
                ['xsel', '-b']]
    elif sys.platform.startswith('win'):
        cmds = [['powershell', '-noprofile', '-command', 'Get-Clipboard']]
    else:
        cmds = []

    for cmd in cmds:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=5)
            if out.returncode == 0:
                return out.stdout.rstrip('\r\n')
        except Exception:
            continue
    return None


def paste_button(text_w: W.Widget, *, append: bool = False,
                 tooltip: str = 'Paste from system clipboard',
                 width: str = '34px') -> W.Button:
    """A small 📋 button that fills ``text_w.value`` from the system
    clipboard on click.  ``append=True`` adds to the existing value
    instead of replacing it.  No-op (with a brief ✗) if the clipboard
    can't be read (e.g. remote kernel with no backend)."""
    btn = W.Button(description='📋', tooltip=tooltip,
                   layout=W.Layout(width=width, padding='0px'))

    def _on_click(_b):
        text = read_system_clipboard()
        if text is None:
            btn.description = '✗'
            btn.tooltip = ('Could not read the clipboard.  Install '
                           '`pyperclip` (pip install pyperclip) or run a '
                           'local kernel.')
            return
        text_w.value = (text_w.value + text) if append else text
        btn.description = '📋'   # clear any prior ✗

    btn.on_click(_on_click)
    return btn


# ── Scalar / vector / matrix value editors ────────────────────────────

def vector_input(label: str, length: int,
                 defaults: Optional[list[float]] = None,
                 width: str = '70px') -> W.HBox:
    """A horizontal row of ``FloatText`` widgets representing a vector.

    Returns an ``HBox``; the underlying widgets are exposed as
    ``box.children[1:]`` (the first child is the label).
    The composite carries a ``.get_value()`` method returning a
    Python list.
    """
    label_w = W.Label(label, layout=W.Layout(width='100px'))
    items = []
    for j in range(length):
        v = (defaults[j] if defaults is not None and j < len(defaults)
             else 0.0)
        items.append(W.FloatText(value=float(v),
                                 layout=W.Layout(width=width)))
    box = W.HBox([label_w] + items)
    box.get_value = lambda: [it.value for it in items]
    box._items = items   # for external setters
    return box


def matrix_input(label: str, n_rows: int, n_cols: int,
                 defaults: Optional[list[list[float]]] = None,
                 width: str = '70px') -> W.VBox:
    """An N×M grid of ``FloatText`` widgets.

    Returns a ``VBox`` with ``.get_value()`` returning a list of lists.
    """
    label_w = W.HTML(f'<b>{label}</b>')
    rows = []
    cells = []
    for i in range(n_rows):
        row_cells = []
        for j in range(n_cols):
            v = 0.0
            if defaults and i < len(defaults) and j < len(defaults[i]):
                v = defaults[i][j]
            row_cells.append(W.FloatText(value=float(v),
                                         layout=W.Layout(width=width)))
        cells.append(row_cells)
        rows.append(W.HBox(row_cells))
    box = W.VBox([label_w] + rows)
    box.get_value = lambda: [[c.value for c in row] for row in cells]
    box._cells = cells
    return box


# ── Validating text input ─────────────────────────────────────────────

def expression_input(label: str, value: str = '',
                     placeholder: str = '',
                     validator: Optional[Callable[[str], tuple[bool, str]]] = None,
                     width: str = '300px') -> W.HBox:
    """A single-line text input with a status icon for live validation.

    ``validator(text)`` returns ``(ok: bool, message: str)``.  If
    ``ok``, the indicator turns green (✓); otherwise red (✗) with the
    message as a tooltip.

    The ``HBox`` carries a ``.get_value()`` method returning the
    current text.
    """
    label_w = W.Label(label, layout=W.Layout(width='80px'))
    text_w  = W.Text(value=value, placeholder=placeholder,
                     layout=W.Layout(width=width))
    indicator = W.HTML(value='', layout=W.Layout(width='30px'))

    def _update(change=None):
        s = text_w.value
        if not s.strip():
            indicator.value = ''
            return
        if validator is None:
            indicator.value = "<span style='color:#888'>?</span>"
            return
        try:
            ok, msg = validator(s)
        except Exception as e:
            ok, msg = False, str(e)
        if ok:
            indicator.value = "<span style='color:#27AE60' title='ok'>✓</span>"
        else:
            indicator.value = (f"<span style='color:#E74C3C' "
                               f"title='{msg}'>✗</span>")

    text_w.observe(_update, names='value')
    _update()

    box = W.HBox([label_w, text_w, paste_button(text_w), indicator])
    box.get_value = lambda: text_w.value
    box._text_w = text_w
    return box


def textarea_input(label: str, value: str = '',
                   placeholder: str = '',
                   rows: int = 8,
                   width: str = '600px') -> W.VBox:
    """A multi-line text area (used by the action editor).  The label
    row carries a 📋 paste button (VS Code's notebook renderer blocks
    Ctrl/Cmd-V into text inputs; the button reads the clipboard
    Python-side instead)."""
    label_w = W.HTML(f'<b>{label}</b>')
    text_w  = W.Textarea(value=value, placeholder=placeholder,
                         layout=W.Layout(width=width, height=f'{rows*18}px'))
    header = W.HBox([label_w, paste_button(text_w)])
    box = W.VBox([header, text_w])
    box.get_value = lambda: text_w.value
    box._text_w = text_w
    return box


# ── Dynamic table (add / remove rows) ────────────────────────────────

class DynamicTable:
    """A spreadsheet-like editor: each row is a dict of named values,
    each column declared up front via ``columns``.

    Parameters
    ----------
    columns : list of dict
        Each dict has keys::
            {'name':              'name',
             'kind':              'text' | 'bool' | 'select' | 'int' | 'float',
             'options':           [...],          # static options for 'select'
             'options_provider':  callable () -> list,
                                  # dynamic options — re-queried on demand
                                  # (use .refresh_dropdown_options to push
                                  # an update after the underlying data changes)
             'default':           <value>,
             'width':             '120px'}
    initial : list of dict, optional
        Pre-populate rows with these values.

    Methods
    -------
    .show()       → returns the ipywidgets layout
    .get_rows()   → returns list of dicts (one per row)
    .clear()      → remove all rows
    .add_row(values=None)
    .on_change(callback)  → fire callback on any add/remove/cell-edit
    .refresh_dropdown_options(col_name)
                  → re-query the column's ``options_provider`` and
                    push the new options onto every existing row's
                    dropdown.  Preserves prior selections when still valid.
    """

    def __init__(self, columns: list[dict],
                 initial: Optional[list[dict]] = None,
                 add_label: str = '+ add row'):
        self._columns = columns
        self._row_widgets: list[dict] = []   # list of {col_name: widget}
        self._row_boxes:   list[W.HBox] = []  # one HBox per row, for layout
        self._change_callbacks: list[Callable[[], None]] = []

        self._header = W.HBox([
            W.HTML(f"<b style='width:{c.get('width', '120px')};'>{c['name']}</b>",
                   layout=W.Layout(width=c.get('width', '120px')))
            for c in columns
        ] + [W.HTML("<b style='width:80px;'></b>")])

        self._rows_container = W.VBox([])
        self._add_btn = W.Button(description=add_label,
                                 button_style='info',
                                 layout=W.Layout(width='160px'))
        self._add_btn.on_click(lambda _: (self.add_row(), self._notify_change()))

        if initial:
            for row in initial:
                self.add_row(values=row, _notify=False)
        else:
            self.add_row(_notify=False)

    def _make_widget(self, col: dict, value: Any = None) -> W.Widget:
        kind = col.get('kind', 'text')
        width = col.get('width', '120px')
        layout = W.Layout(width=width)
        default = col.get('default')
        v = value if value is not None else default

        if kind == 'text':
            w = W.Text(value='' if v is None else str(v),
                       placeholder=col.get('placeholder', ''),
                       layout=layout)
            w.observe(lambda _change: self._notify_change(), names='value')
            return w
        if kind == 'bool':
            w = W.Checkbox(value=bool(v) if v is not None else False,
                           indent=False, layout=layout)
            w.observe(lambda _change: self._notify_change(), names='value')
            return w
        if kind == 'select':
            # Static options vs. dynamic via options_provider.
            provider = col.get('options_provider')
            opts = provider() if provider is not None else col.get('options', [])
            initial = v if v in opts else (opts[0] if opts else None)
            w = W.Dropdown(options=opts, value=initial, layout=layout)
            w.observe(lambda _change: self._notify_change(), names='value')
            return w
        if kind == 'int':
            w = W.IntText(value=int(v) if v is not None else 0, layout=layout)
            w.observe(lambda _change: self._notify_change(), names='value')
            return w
        if kind == 'float':
            w = W.FloatText(value=float(v) if v is not None else 0.0,
                            layout=layout)
            w.observe(lambda _change: self._notify_change(), names='value')
            return w
        raise ValueError(f'unknown column kind: {kind!r}')

    def add_row(self, values: Optional[dict] = None,
                _notify: bool = True) -> None:
        widgets = {}
        children = []
        for col in self._columns:
            v = (values or {}).get(col['name'])
            w = self._make_widget(col, v)
            # The VALUE widget is always ``w`` (so get_rows reads it
            # unchanged).  Columns flagged ``'paste': True`` get a 📋
            # button beside the input — a click-driven clipboard fill
            # that works around VS Code swallowing Ctrl/Cmd-V.
            widgets[col['name']] = w
            if col.get('paste') and col.get('kind', 'text') == 'text':
                children.append(W.HBox([w, paste_button(w, width='30px')]))
            else:
                children.append(w)
        # Remove button
        rm_btn = W.Button(description='✕', button_style='warning',
                          layout=W.Layout(width='40px'))

        def _make_remove(_idx):
            def _on_click(_btn):
                if 0 <= _idx < len(self._row_widgets):
                    for i, rb in enumerate(self._row_boxes):
                        if rb.children[-1] is _btn:
                            self._row_widgets.pop(i)
                            self._row_boxes.pop(i)
                            self._rows_container.children = tuple(
                                self._row_boxes)
                            self._notify_change()
                            return
            return _on_click
        rm_btn.on_click(_make_remove(len(self._row_widgets)))

        children.append(rm_btn)
        row_box = W.HBox(children)
        self._row_widgets.append(widgets)
        self._row_boxes.append(row_box)
        self._rows_container.children = tuple(self._row_boxes)
        if _notify:
            self._notify_change()

    def clear(self) -> None:
        self._row_widgets = []
        self._row_boxes = []
        self._rows_container.children = ()
        self._notify_change()

    def get_rows(self) -> list[dict]:
        out = []
        for w_dict in self._row_widgets:
            row = {name: w.value for name, w in w_dict.items()}
            # Drop empty rows (no name)
            if 'name' in row and not str(row['name']).strip():
                continue
            out.append(row)
        return out

    def show(self) -> W.VBox:
        return W.VBox([self._header, self._rows_container, self._add_btn])

    # ── Change-event plumbing ──────────────────────────────────────
    def on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback fired whenever a row is added, removed,
        or a cell value changes.  Multiple callbacks may be registered."""
        self._change_callbacks.append(callback)

    def _notify_change(self) -> None:
        for cb in self._change_callbacks:
            try:
                cb()
            except Exception:
                pass

    def refresh_dropdown_options(self, col_name: str) -> None:
        """Re-query the ``options_provider`` for the named ``select``
        column and update every existing row's dropdown.  Preserves
        prior selections when still valid; otherwise falls back to
        the first option."""
        col = next((c for c in self._columns if c['name'] == col_name), None)
        if col is None:
            return
        provider = col.get('options_provider')
        if provider is None:
            return
        new_opts = list(provider())
        for w_dict in self._row_widgets:
            w = w_dict.get(col_name)
            if not isinstance(w, W.Dropdown):
                continue
            prev = w.value
            # Setting .options while value is invalid raises; clear first.
            try:
                w.options = new_opts
                if prev in new_opts:
                    w.value = prev
                elif new_opts:
                    w.value = new_opts[0]
            except Exception:
                # Some ipywidgets versions need a different ordering.
                w.value = (new_opts[0] if new_opts else None)
                w.options = new_opts

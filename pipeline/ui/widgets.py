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

from typing import Any, Callable, Optional

import ipywidgets as W


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

    box = W.HBox([label_w, text_w, indicator])
    box.get_value = lambda: text_w.value
    box._text_w = text_w
    return box


def textarea_input(label: str, value: str = '',
                   placeholder: str = '',
                   rows: int = 8,
                   width: str = '600px') -> W.VBox:
    """A multi-line text area (used by the action editor)."""
    label_w = W.HTML(f'<b>{label}</b>')
    text_w  = W.Textarea(value=value, placeholder=placeholder,
                         layout=W.Layout(width=width, height=f'{rows*18}px'))
    box = W.VBox([label_w, text_w])
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
            {'name':    'name',
             'kind':    'text' | 'bool' | 'select' | 'int' | 'float',
             'options': [...],          # for 'select'
             'default': <value>,
             'width':   '120px'}
    initial : list of dict, optional
        Pre-populate rows with these values.

    Methods
    -------
    .show()       → returns the ipywidgets layout
    .get_rows()   → returns list of dicts (one per row)
    .clear()      → remove all rows
    .add_row(values=None) → append a row with given/default values
    """

    def __init__(self, columns: list[dict],
                 initial: Optional[list[dict]] = None,
                 add_label: str = '+ add row'):
        self._columns = columns
        self._row_widgets: list[dict] = []   # list of {col_name: widget}
        self._row_boxes:   list[W.HBox] = []  # one HBox per row, for layout

        self._header = W.HBox([
            W.HTML(f"<b style='width:{c.get('width', '120px')};'>{c['name']}</b>",
                   layout=W.Layout(width=c.get('width', '120px')))
            for c in columns
        ] + [W.HTML("<b style='width:80px;'></b>")])

        self._rows_container = W.VBox([])
        self._add_btn = W.Button(description=add_label,
                                 button_style='info',
                                 layout=W.Layout(width='160px'))
        self._add_btn.on_click(lambda _: self.add_row())

        if initial:
            for row in initial:
                self.add_row(values=row)
        else:
            self.add_row()

    def _make_widget(self, col: dict, value: Any = None) -> W.Widget:
        kind = col.get('kind', 'text')
        width = col.get('width', '120px')
        layout = W.Layout(width=width)
        default = col.get('default')
        v = value if value is not None else default

        if kind == 'text':
            return W.Text(value='' if v is None else str(v),
                          placeholder=col.get('placeholder', ''),
                          layout=layout)
        if kind == 'bool':
            return W.Checkbox(value=bool(v) if v is not None else False,
                              indent=False, layout=layout)
        if kind == 'select':
            opts = col.get('options', [])
            initial = v if v in opts else (opts[0] if opts else None)
            return W.Dropdown(options=opts, value=initial,
                              layout=layout)
        if kind == 'int':
            return W.IntText(value=int(v) if v is not None else 0,
                             layout=layout)
        if kind == 'float':
            return W.FloatText(value=float(v) if v is not None else 0.0,
                               layout=layout)
        raise ValueError(f'unknown column kind: {kind!r}')

    def add_row(self, values: Optional[dict] = None) -> None:
        widgets = {}
        children = []
        for col in self._columns:
            v = (values or {}).get(col['name'])
            w = self._make_widget(col, v)
            widgets[col['name']] = w
            children.append(w)
        # Remove button
        rm_btn = W.Button(description='✕', button_style='warning',
                          layout=W.Layout(width='40px'))

        def _make_remove(_idx):
            def _on_click(_btn):
                if 0 <= _idx < len(self._row_widgets):
                    # Find the actual current index (rows may shift)
                    for i, rb in enumerate(self._row_boxes):
                        if rb.children[-1] is _btn:
                            self._row_widgets.pop(i)
                            self._row_boxes.pop(i)
                            self._rows_container.children = tuple(
                                self._row_boxes)
                            return
            return _on_click
        rm_btn.on_click(_make_remove(len(self._row_widgets)))

        children.append(rm_btn)
        row_box = W.HBox(children)
        self._row_widgets.append(widgets)
        self._row_boxes.append(row_box)
        self._rows_container.children = tuple(self._row_boxes)

    def clear(self) -> None:
        self._row_widgets = []
        self._row_boxes = []
        self._rows_container.children = ()

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

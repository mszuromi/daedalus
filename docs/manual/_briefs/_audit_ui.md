# Audit — `sections/21-ui.tex` (The Theory-Builder UI)

**Verdict: accurate (minor-issues).** Every named symbol, file, and line citation
in the chapter was verified against `pipeline/ui/main.py`, `pipeline/ui/widgets.py`,
`pipeline/_precompute.py`, and `pipeline/theory_serialize.py`. No invented or
renamed identifiers. No backwards gates. No wrong-file citations. LaTeX compiles
(all environments balanced; no bare specials outside math/listings; all macros and
referenced chapter labels defined; chapter is `\input`-ed at
`daedalus_manual.tex:219`).

The findings below are all **minor / nit** — small overstatements and prose/format
slips. Nothing requires a correction for correctness of the physics or the API.

## Findings

- **minor** — `_make_remove` captured index called "effectively dead code" / "The
  handler ignores it". The captured `_idx` IS used: `widgets.py:331` guards the
  removal with `if 0 <= _idx < len(self._row_widgets):` before the identity scan.
  Because `_idx` is captured (as the pre-append count) at row-creation and the list
  shrinks as rows are removed, this guard can genuinely evaluate False for a
  high-index row after earlier deletions — so it has observable effect, it is not
  dead code. The chapter's load-bearing claim (the *identity scan*, not the index,
  selects *which* row to pop) is correct; only "ignores it / effectively dead code"
  overstates. (chapter L452–458 vs `pipeline/ui/widgets.py:329–341`)

- **nit** — Column-spec key list attributed entirely to the class docstring ("The
  recognized keys, from the class docstring (widgets.py:218)"), but the chapter's
  listing adds `'placeholder'` and `'paste'`, which are NOT in the docstring
  (`widgets.py:219–228`); they are only read in code (`col.get('placeholder',...)`
  at `widgets.py:281`, `col.get('paste')` at `widgets.py:321`). The keys are real
  recognized keys — only the "from the docstring" attribution is imprecise.
  (chapter L418–429 vs `pipeline/ui/widgets.py:218–228`)

- **nit** — Duplicated words in prose: "...is a single-line text box with a live / a
  live check-mark / cross-mark indicator..." — "a live" appears twice across the
  line break. Pure prose defect. (chapter L388–389)

- **nit** — `sum` described as one of the "indexing helpers" in `_ACTION_BUILTINS`.
  Set membership is correct (`sum` IS in the frozenset), but in the source `sum`
  sits under the "Standard math" comment (`main.py:1655`); only `range`, `len` are
  under "Indexing helpers" (`main.py:1659–1660`). Grouping label only.
  (chapter L746 vs `pipeline/ui/main.py:1651–1661`)

- **nit** — Matrix autofill template shown as `'[[, , ],[, , ]]'`; the code emits
  `'[[, , ], [, , ]]'` (a space after the inter-row comma, from `', '.join`).
  Trivial whitespace in an illustrative code-font example.
  (chapter L575 vs `pipeline/ui/main.py:358–360`)

- **nit** — "`paste_button` ... handles the `None` case by briefly setting its glyph
  to a cross mark". The ✗ glyph is set on a failed read and is only reset to 📋 on a
  later *successful* click (`widgets.py:82–88`); it is not auto-reverted, so
  "briefly" is slightly imprecise. (chapter L375–377 vs `pipeline/ui/widgets.py:79–88`)

- **nit** — "The action validator knows about all three (main.py:1684)". Line 1684 is
  only the physical field itself (`names.add(nm)`); the three auto-companions
  (`{nm}t`, `d{nm}`, `{nm}star`) are added at `main.py:1685–1687`. Citation points one
  line above the trio it references — off by a line, same region.
  (chapter L270 vs `pipeline/ui/main.py:1684–1687`)

## Spot-checks that PASSED (high-value, non-obvious)

- `severity ∈ {warn, error, info}` (chapter L774) is CORRECT despite the `_validate`
  docstring saying `{'warn','error'}` — `info` is actually emitted at
  `main.py:1952`, and the chapter's "info does not trip a badge" matches the badge
  logic (`main.py:2114–2122`) and the code comment (`main.py:2134–2138`). The chapter
  is more accurate than the code's own stale docstring. NOT a finding.
- `show()` listing, `_CSS_INJECTED` guard (244), `_THEORY_BUILDER_CSS` namespace (63),
  `spec()` (2330) — all exact.
- All ten tab widget names (`_w_name`…`_tbl_ext_fields`), `_all_tabs` order (1317),
  `W.Tab` numbering, toggle options `['Temporal (ODE)','Spatial (PDE)']` (404) — exact.
- Action placeholder `phit * ((Dt + mu) * phi + eps * phi^3) - D * phit^2` at
  `main.py:1013` — exact.
- `_ACTION_BUILTINS` membership (1651), operator-IR adds `Lap/Dx/Gradient/GradX` only
  when ticked (1712–1713), spatial adds `Laplacian` (1711) — exact.
- Reserved names `{t,omega,Dt,delta_D,delta_Dp}` + spatial `{k,Laplacian,x,y,z}`
  (1915/1917) — exact.
- per-field equation check is word-boundary regex, fires only for ≥2 fields
  (1991–2006) — exact.
- `_collect` spec-dict shape (2498), metadata dict (2570), `fixed_point_index_default`
  always written (2620), `default_fundamental` always empty (2544),
  `eval(text,{'__builtins__':{}},{})` for params (2342, 2427) vs `ast.literal_eval`
  for seed box (2586) — exact.
- Mode-switch: Temporal force-zeros + clears spatial knobs (1534–1543); Spatial clears
  Kernels+Noise (1548–1549); `_apply_visible_tabs` hides Kernels+Noise tabs (1560) — exact.
- Bottom-bar handlers `_on_preview` (2640), `_on_save` (2650), `_on_reset` (2689),
  `_on_precompute` (2704), `_on_open_in_runner` (2231), `_on_load` (2769), `load` (2813),
  `_mark_changed` (2204), `_check_dirty_guard` (2217) — all exact.
- Pre-compute sequence (`render_theory_file → exec/compile → ns['build']() →
  precompute(model, verbose=True)`, 2727–2740) and `precompute` in `pipeline/_precompute.py`
  (Taylor order 2, mf_check, propagator_built, cache_dir, wall_seconds) — exact.
- `.theories/.last_built` scratch file + `Javascript("window.open('theory_runner.ipynb',
  '_blank');")` (2253–2259) — exact.
- Clipboard: `read_system_clipboard` (29) pyperclip→pbpaste/wl-paste/xclip/xsel/
  Get-Clipboard, `shutil.which` skip, `subprocess.run(...,timeout=5)`; `paste_button`
  (69) — exact.
- `DynamicTable.__init__(columns, initial=None, add_label='+ add row')` (245),
  `_make_widget` dispatch + `ValueError` on unknown kind (272/307), `get_rows`
  blank-`name` drop (365), `set_column_visible` (375), `refresh_dropdown_options`
  try/except ordering (410), `_notify_change` silent-except (403) — exact.
- Bulk import at `main.py:30`: only `textarea_input`, `DynamicTable`, `paste_button`
  actually called; `vector_input`/`matrix_input`/`expression_input` import-only — verified.
- `theory_serialize.py:455` "silently loses the Dt()/Lap()/Dx() lowering" comment — exact.
- Module docstring "9-section tab editor" (main.py:4) is stale; chapter correctly flags
  it and trusts `_all_tabs` (10) — exact.
- File sizes: chapter says widgets.py "(438 lines)" / "(438 lines)" — actual 437 (last
  line likely blank-terminated). Within ±1; not flagged.

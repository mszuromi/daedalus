# Audit — `sections/02-quickstart.tex`

**Verdict: minor-issues** (3 minor/nit findings; no major/critical). LaTeX: **valid** (compiles clean).

Scope: every named identifier, file:line citation, behavioural claim, code listing, and
numeric result in the chapter was checked against `notebooks/daedalus.py`,
`pipeline/compute.py`, `theories/ou_quartic.theory.py`, the example notebook
`notebooks/examples/temporal_ou_quartic_white.ipynb`, plus the `msrjd` package internals.

---

## Findings

- **[minor] "two-line digest" is actually three lines.**
  - Manual claim (line 393-394): *"the run is quiet and `summary` prints a **two-line** digest"*, then quotes three lines of output (`theory : …`, `k : … max_ell : …`, `fields : … spatial_dim : …`).
  - Code reality: `summary()` builds exactly three lines (`daedalus.py:1091-1094`), and the notebook's stored cell-7 output is three lines. The quoted block is correct; only the word "two-line" is wrong (should be "three-line").
  - Location: `02-quickstart.tex:393-400` vs `notebooks/daedalus.py:1091-1094`.

- **[nit] Step-5 grid citation points into the spatial branch, not the temporal one.**
  - Manual claim (line 377-379): *"Wire the temporal grid (`notebooks/daedalus.py:578`): `tau_max` and `tau_step` from the `Config` (or `METADATA`)."*
  - Code reality: line 578 is inside the `if is_spatial(model):` block (`kw['tau_max'] = 0.0 if cfg.tau_max is None else cfg.tau_max`). The **temporal** grid wiring this sentence describes is the `else:` branch at `daedalus.py:580-584` (`kw['tau_max'] = _meta(module,'tau_max',10.0) if cfg.tau_max is None else cfg.tau_max`). Same function, ~3 lines off, wrong branch. Description of the behaviour is otherwise correct.
  - Location: `02-quickstart.tex:377-379` vs `notebooks/daedalus.py:578` (cited) / `:580-584` (correct).

- **[nit] `res['_resolved']` key list is incomplete.**
  - Manual claim (line 387-389): the `_resolved` dict records "the resolved `k`, `max_ell`, `external_fields`, and `parameters`" (4 keys).
  - Code reality (`daedalus.py:696-697`): `dict(k=k, max_ell=max_ell, external_fields=ext, parameters=fundamental, fundamental=fundamental)` — a 5th key `fundamental` (a backward-compat alias of `parameters`) is also present. The manual's list is not wrong, just not exhaustive; and `res['_resolved']['parameters']` (used in the sim step) does exist (`:697`).
  - Location: `02-quickstart.tex:387-389` vs `notebooks/daedalus.py:696-697`.

---

## Verified accurate (high-signal items that were specifically checked)

**Identifiers — all exist, none invented/renamed:**
- `load_theory` (`daedalus.py:68`, three-step importlib idiom returns `mod.build(), mod` — `:77-80`, cited `:77` exact), `list_theories` (`:61`), `repo_root` (`:40`, cited exact; walks up for `pipeline/`), `describe_model` (`:138`), `field_names` (`:105`), `is_spatial` (`:85`), `Config` (`@dataclass`, `:247-248`), `run` (`:487`, cited exact), `summary` (`:1087`, cited exact), `plot_cumulant` (`:725`).
- `compute_cumulants`, `compute_correction_td` (real def `msrjd/integration/time_domain/pipeline.py`; called `compute.py:776`), `FieldTheory.expand` (`msrjd/core/field_theory.py:814`), `build_propagator`/`compute_poles_and_residues` (`pipeline/_propagator.py`), `enumerate_unique_diagrams` (`pipeline/_diagrams.py`), `TemporalTheoryBuilder` (`pipeline/theory.py:2087`).
- Simulator imports both exist: `sim_ou_quartic_numba` (`models/ou_langevin_sim_numba.py:18`), `estimate_kpoint_slices` (`models/cumulant_estimator.py:445`).

**Field-naming claims (the subtle ones) — all correct:**
- physical field declared `'x'` → internal leg `dx`, so `field_names(model)` returns `['dx']` (new-style `physical_field`, `theory.py:858-878`). Confirmed by live notebook output `physical fields: ['dx']`.
- response field `xt` (`<natural>t`), saddle param `xstar` (`<natural>star`). `describe_model` shows `Fields: x` via `natural_name` (`:168`). All match the notebook's cell-4 output.

**Seven-phase trace — every label string and every cited line is exact:**
`[1/7]`→`compute.py:312`, `[2/7]`→`:355`, `[3/7]`→`:362`, `[4/7]`→`:630`, `[5/7]`→`:637`, `[6/7]`→`:668`, `[7/7]`→`:709` (renders `(0..2)` with max_ell=2 — matches). `Done.` line `:947-949`. Spatial-divergence-after-phase-3 note supported by `compute.py:488-490`. Wall-time breakdown block `:950-953` (chapter cites `:947`, same print block — fine).

**Behavioural claims — verified against source:**
- `run` loop-order resolution `cfg.max_ell else ell_default else 0` (`:494`, cited exact). k-resolution raises `ValueError` on k/leg mismatch (`:504-511`). Explicit `external_fields` used verbatim; fallback only when unset/stale (`:517-533`, cited `:517` exact). Parameter layering: model defaults ← `DEFAULT_FUNDAMENTAL` ← `cfg.parameters` (`:538-542`, cited exact). `res = compute_cumulants(**kw)` (`:586`, cited exact). Stamps `_cfg`/`_model`/`_resolved` (`:694-697`).
- `Config`: `output` default `'cumulant'`; `'moment'/'central_moment'` cost `k−1` extra runs (`:265-266`). `fundamental` deprecated alias; `__post_init__` mirrors, `parameters` wins (`:312-319`). `parallel=False` default (`:301`).
- Result dict: `tau_grid`/`C_tau`/`C_tau_by_ell`/`total_C`/`total_C_by_ell` all present (`compute.py:882-886`). `C_tau_by_ell[ell]` is **incremental** per-order (`:791-795`); `C_tau = Σ_ell C_tau_by_ell[ell]` (`:808`); `total_C` callable sums across ell (`:802-804`). So `C0_tree` (ell 0) vs `C0_loop` (full sum) labelling is correct. Plot draws **cumulative** curves by default (`show_orders='cumulative'`, `:305`; `cumulative_curves`, `:713-714`).
- `describe_model` prints AND returns the string (`:241-242`); strips changelog/provenance via `_DOC_DROP`/`_DOC_STOP` (`:116-135`). `plot_cumulant` takes optional `sim` with `tau`/`C`/`C_err` keys (`:726`, docstring `:735-736`).
- τ grid is symmetric `[-tau_max, tau_max]` step `tau_step` (`compute.py:712`) → `[-8,8]` step 0.5, matching the chapter.

**Code listings & numbers — faithful to the notebook:**
- Setup cell (`02-quickstart.tex:81-92`) matches notebook cell 2 verbatim (incl. the `# cwd=notebooks/ …` comment). Config cell, run cell, C0-extraction cell, and sim cell all match notebook cells 6/7/9 (including `dt_sim=0.01, dt_bin=0.02, T_sim=2e5, N_RUNS=3`).
- `theories/ou_quartic.theory.py` listing matches the real file (build chain, `DEFAULT_FUNDAMENTAL`, `METADATA`) — only the docstring is omitted (acceptable for a `build()` excerpt).
- Headline numbers exactly match notebook stored output: `tree = 1.0000   tree+loops = 0.9496   sim = 0.9464 ± 0.0009`, `sim took 7.7s`. `g_eff = εD/μ² = 0.02` arithmetic correct. "~0.003 gap ≈ 3σ" (0.0032/0.0009 ≈ 3.6) reasonable.
- `describe_model` quoted output matches notebook cell 4 verbatim (Domain/Fields/Response/saddle `xstar`/Governing eqn/Suggested run/docstring).

---

## LaTeX validity → **latex_ok = true**

- Master build compiles **clean**: `daedalus_manual.log` (built 14:44 Jun 18, newer than the chapter's 00:05 mtime) has **zero** `!` error lines, **zero** "Undefined control sequence", and processed `(./sections/02-quickstart.tex [22]` into a 406-page PDF. (465 Overfull/Underfull box warnings across the whole manual — cosmetic only.)
- Every macro/environment used in the chapter is defined in the preamble: `\code`,`\file`,`\term` (`daedalus_manual.tex:98-100`), `\msrjd` (`:102`), `\avg`/`\ee`/`\Gret`/`\Sym` (`:119-124`), `note`/`gotcha` tcolorboxes (`:107-110`), `lstdefinestyle{console}` (`:83`). `\Gret` renders `G_{\mathrm R}` — a math macro, not a code symbol, used correctly for `G_ft`.
- No bare specials in prose: every `_` is inside `$…$` math or an `lstlisting`/`console` body or written `\_` inside `\code{}`; every `^` is math-mode or escaped `\^{}` (lines 207-208); the lone text `%` is `\%` (line 515); no bare `&`/`#` in prose (all inside listings/tikz comments). The `\dd` math macro (preamble `:117`) is **not** shadowed — the module alias `dd` always appears as literal text inside `\code{}`, never as `\dd`.
- The `tikzpicture` sketch (lines 610-629) is balanced and uses only `arrows.meta`/`calc`-compatible syntax already loaded in the preamble.

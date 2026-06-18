# Audit — `sections/18-engine-api.tex` (The `daedalus` Engine API)

**Source audited against:** `notebooks/daedalus.py` (1101 lines)
**Verdict:** accurate (one nit) · **latex_ok:** true

## Summary

The chapter is an unusually faithful, line-cited reading of `notebooks/daedalus.py`.
Every named function, class, constant, `Config` field, dispatch branch, plotter,
and behavioural claim was verified against the real source by targeted grep/Read.
The chapter compiles cleanly as Chapter 18 (PDF p.293+) with zero LaTeX errors and
all internal/cross-chapter `\ref`/`\eqref`/`\label` targets resolving.

## Verification highlights (all PASS)

- **Top-level symbols** all exist at the cited lines: `repo_root`:40, `list_theories`:61,
  `load_theory`:68, `is_spatial`:85, `spatial_dim`:90, `is_multifield`:95, `field_names`:105,
  `_fmt_default`:109, `_doc_physics`:125, `describe_model`:138, `Config`:248(decl 247),
  `_meta`:331, `fundamental_from_model`:335, `parameters_from_model`:352 (alias),
  `_set_partitions`:357, `_external_mean`:373, `_assemble_moment_temporal`:390, `run`:487,
  `_ORDER_COLORS`:703, `_order_label`:707, `cumulative_curves`:713, `plot_cumulant`:725,
  `_plot_moment`:759, `_orders_to_draw`:801, `plot_temporal`:815,
  `plot_temporal_kpoint_slices`:856, `plot_temporal_kpoint_grid`:923,
  `plot_temporal_kpoint`:948, `plot_spatial`:1000, `plot_kpoint`:1054, `summary`:1087.
- **Config table** (`tab:config`): every field/default/line citation (:255–:310) is correct,
  in declaration order, including `output='cumulant'` default and the deprecated `fundamental`
  alias semantics (`parameters` wins, :312–319).
- **`run` control flow**: loop-order resolve (:494–495), k-resolution + contradiction guard
  (:499–515, quoted block :506–511), external-leg auto-build (:517–533), 3-layer `fundamental`
  (:534–542), in-place Dyson mutation (:545–550), MF-root forwards only-when-set with engine
  defaults `fixed_point_index=0`/`mf_dae_n_starts=64`/`mf_dae_seed_box=None` (:555–564), kw
  assembly (:552–553), spatial-vs-temporal grid wiring incl. spatial k=2 defaults
  `tau_max=0.0`/`tau_step=1.0` and temporal `10.0`/`0.5` (:566–584), engine call (:586),
  result stamps `_cfg`/`_model`/`_resolved` (:694–697). All accurate.
- **k≥3 synthesis**: 41-pt cap (:601–602), base validation/default (:608–616), `_args`/`_slice`
  (:618–630), `C_tau=slices[1]` canonicalisation (:641–642), full-grid downsample target
  `min(tau.size, max(2, int(4000**(1/(k-1)))))` (:652), `try/except Exception: pass` swallow
  (:632, :671–672). All accurate.
- **Moments**: shared-loop-budget formula matches `_assemble_moment_temporal` exactly, all
  6 step citations (:411–414, :416–429, :431–436, :438–448, :452–458, :460–483) correct;
  `_set_partitions` Bell-number behaviour correct; spatial k=2 one-liner + `μ²` raw term
  (:681–684) and spatial k≥3 `NotImplementedError` (:685–688) correct.
- **Dispatch tree** (:738–756): all 7 ordered branches verified in order
  (moment → C_kpoint → spatial → C_tau-None → C_tau_grid → ≥2 slices → plot_temporal).
- **Seven plotters + dispatcher**: grep confirms exactly 8 `plot_/_plot_` defs = 7 plotters +
  1 dispatcher; the chapter's "seven plotters" count and per-plotter descriptions
  (axhline/axhspan scalar sim, argmin(|tau|) row, `spatial_info['C_by_order']`, grouped bars,
  no-`sim`-arg `plot_kpoint`) all match.
- **Gotchas section**: every one holds —
  in-place model mutation (:545–550); fatal k/legs mismatch (:506–511); verbatim explicit
  `external_fields` (:519–533); spatial k≥3 needs `spatial_points` + moments NotImplemented;
  exception swallow; 41-pt/downsample caps; tree-mean tadpole approximation;
  `Config.components` declared (:307) + advertised (:23–25) but **never read** by any plotter
  (confirmed by scanning all plotters); `is_multifield` exists but dispatcher never branches on
  it; `from dataclasses import … field` (:31) **unused** (confirmed: zero `field(` call sites).
- **"daedalus does almost no physics"**: confirmed — the only non-stdlib imports are `numpy`
  and `matplotlib.pyplot`; no sage/nauty/sympy/numba/networkx/scipy token appears anywhere in
  the file; the heavy work is behind `from pipeline import compute_cumulants` (:410, :492).
- **LaTeX**: all macros (`\code`,`\file`,`\term`,`\msrjd`,`\avg`, tcolorboxes `note`/`gotcha`/
  `defn`) defined in preamble; `codebg`/`notebg` colours, `console` lst style, tikz
  arrows.meta/positioning, longtable/booktabs/amsmath all loaded. Log shows **no `!` errors**
  for this chapter; all ch18 labels present in `.aux`; all cross-chapter refs
  (quickstart/theory-spec/theory-files/mean-field/spatial-coupled/cumulants-moments) resolve.
  (The `sec:parallel`/`eq:GR`/`sec:dispatch` "multiply defined" warnings in the log belong to
  OTHER chapters, not ch18.)

## Findings

- **nit** — *manual claim:* the `_external_mean` "1-loop tadpole shift of the mean is not yet
  folded in" refinement is cited at `notebooks/daedalus.py:376-377`.
  *code reality:* that sentence actually occupies lines **375–376** (`"""…The 1-loop tadpole
  shift of / the mean is not yet folded in (a documented refinement)."""`); line 377 is the
  next statement (`mfv = res.get('mf_values') or {}`). Off by one line at the range boundary;
  the cited content is exactly as described, and the companion citation `:404-405` for the same
  refinement in `_assemble_moment_temporal` is exact.
  *location:* §"Cumulants to moments: the set-partition expansion" (enumerate item 2) and
  Gotcha #7, both citing `:376-377`.

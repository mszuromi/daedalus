# Audit — `sections/19-cumulants-moments.tex`

**Verdict: accurate (minor-issues).** One minor numeric-prose imprecision; one cosmetic
docstring-reference mismatch. No invented symbols, no behavioural overstatements, no
wrong-file citations, LaTeX compiles.

Sources cross-checked: `pipeline/compute.py`, `notebooks/daedalus.py`.

## Symbol verification (all PASS)
Every named identifier exists verbatim in the source:
- `compute_cumulants` (compute.py:108), `run` (daedalus.py:487), `plot_cumulant` (725),
  `_plot_moment` (759), `_assemble_moment_temporal` (390), `_set_partitions` (357),
  `_external_mean` (373), nested `_by_ell_2pt` (416), `_kappa` (452), `_args` (618),
  `_slice` (624), `_grid` (656), `field_names` (105),
  `plot_temporal_kpoint_slices` (856).
- Config fields: `output` (266, default `'cumulant'`), `max_ell` (257),
  `kpoint_base_lags` (280), `kpoint_full_grid` (283, default `False`).
- Result keys: `C_tau`, `C_tau_x` (compute.py:588), `C_tau_by_ell`, `total_C` (802),
  `total_C_by_ell`, `C_tau_slices`, `C_tau_slices_by_ell`, `C_tau_grid`,
  `C_tau_grid_by_ell`, `moment`, `output_kind`, `mf_values` (compute.py:596/887),
  `spatial_grid` (compute.py:594).

## File:line citations (all PASS, within stated regions)
- compute.py:108 `compute_cumulants` def — exact.
- compute.py:802 / 808 `total_C` / `C_tau` sum-over-ell — exact (802-804, 808).
- compute.py:807-810 `C_tau=None` for k≥3 — exact (`tau_points=None` set at 735, consumed 807-810).
- daedalus.py:261-266 Config.output comment incl. `M = Σ_π ∏_B κ(B)` and "k−1 extra runs" — exact.
- daedalus.py:357, 373, 390, 416-429, 452-458, 460-483, 466, 468-472 — all exact.
- daedalus.py:588-672 k≥3 synthesis block, 608-616 base validation, 618-630 _args/_slice,
  632/671-672 try/except, 650-670 full grid, 674-692 moment switch — all exact.
- daedalus.py:738-756 dispatch tree, 759 `_plot_moment` — exact; 7-branch order matches
  code 740/743/745/750/752/754/756 line-for-line.
- daedalus.py:856-867 permutation-symmetry note — exact.
- daedalus.py:594 simulator `lag_bins=[0, None, 0, …]` comment — exact.

## Behavioural claims (all PASS)
- Shared-loop-budget partition sum Eq (19.x): `itertools.product(range(L+1), repeat=len(multis))`
  + `if sum(assign) > L: continue` — verified line-for-line (daedalus.py:476-482).
- Central drops singletons (`continue` at 466); raw zero-mean guard (`mu_factor==0.0` at 469);
  all-singleton → `μ^k` (471-472) — all exact.
- `k−1` extra runs: reuse order-k for κ_2 (k==2) and for κ_j when j==k (441-445) — exact.
- k≥3 gate `not is_spatial and k>=3 and C_tau is None and callable(total_C)` (595-596) — exact.
- 41-pt τ cap (601-603); base-point length-validated to k−1 with ValueError (611-614) — exact.
- Full-grid `n = min(tau.size, max(2, int(4000**(1/(k-1)))))` (652); k=3→63→capped 41 — exact.
- Spatial k=2 moment = `Re(C_tau_x) + (μ² if raw)` (681-684); spatial k≥3 → NotImplementedError
  (686-688); temporal → `_assemble_moment_temporal` (690) — exact.
- `_external_mean` candidate-key d-strip (`f0[1:]` if startswith 'd', 380) and 0.0 fallback (387) — exact.
- `_plot_moment`: single curve / no per-order overlay, ylabel `M_k` (789), `argmin(|tau|)` for 2-D
  moment (772), honours logy/title/save (790-797) — exact.
- Tadpole-shift refinement flagged twice: `_external_mean` docstring (374-376) +
  `_assemble_moment_temporal` docstring (403-405). Chapter cites 376-377 and 404-405 — within region.

## Findings

- **nit** — k=4 per-axis grid count. Chapter §"The full grid" (line ~710): "for k=4 it is
  about 4000^{1/3}≈16 points per axis." The code uses `int(4000**(1.0/(k-1)))`
  (daedalus.py:652), and `int(4000**(1/3)) = int(15.87…) = 15`, not 16. The prose quotes the
  un-floored cube root (15.87 ≈ 16) while the actual per-axis count is **15**. (For k=3 the
  chapter's "≈63" is consistent with `int=63`, so the floor convention is applied there but not
  here.) Cosmetic; does not affect any claim about behaviour.

- **nit** — docstring-reference name. Chapter line 152-153 says the identity is written "in the
  `Config.output` comment ... `M = Σ_π ∏_B κ(B)`"; the comment (daedalus.py:264) literally says
  "see `_assemble_moment`" — but the actual function is named `_assemble_moment_temporal`. The
  chapter does NOT repeat this stale name (it correctly calls the function
  `_assemble_moment_temporal` everywhere), so this is a pre-existing source-comment typo the
  chapter faithfully describes around, not a chapter error. Recorded for completeness only.

## LaTeX validity — PASS (`latex_ok = true`)
- All 36 environment pairs balanced (`enumerate`, `equation`, `lstlisting`, `defn`, `example`,
  `note`, `gotcha`, `itemize`, `description`).
- All custom macros defined in preamble `daedalus_manual.tex`: `\code` (98), `\file` (99),
  `\term` (100), `\Daedalus` (101), `\msrjd` (102), `\avg` (122). Theorem-likes:
  `note`/`gotcha`/`defn` tcolorboxes (107/109/111), `example` newtheorem (130).
- No bare `_ # % & $` outside math/listings. The only flagged `_` (lines 141-142) are inside a
  `\[ … \]` display-math block (math subscripts, valid). Lines 1-3 `%` are legitimate comments.
- Chapter is `\input{sections/19-cumulants-moments.tex}` at daedalus_manual.tex:217 — in the build.

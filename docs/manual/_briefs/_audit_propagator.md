# Audit — `sections/07-propagator.tex` (The Propagator)

**Verdict: accurate (minor-issues).** The chapter is an exceptionally faithful
description of `pipeline/_propagator.py` and
`msrjd/integration/time_domain/propagator_td.py`. Every named identifier exists
in the code; every code listing is quoted verbatim (variable names, magic
constants, comments). All custom LaTeX macros/environments are preamble-defined,
all 8 environment types balance (29 `lstlisting`, 14 `gotcha`, 6 `defn`, …),
brace delta = 0, and the chapter compiles cleanly (no `!`-errors in
`daedalus_manual.log`; typeset as page 94 / Chapter 7). **latex_ok = true.**

No CRITICAL or MAJOR findings. No invented/renamed symbols. No behavioural
overstatements that misrepresent control flow. The findings below are all
nit-level (line citations off by a few lines within the correct region) plus one
minor wording softness.

## Verified accurate (spot list)
- `build_propagator` @524, `compute_poles_and_residues` @851, `_horner` @841,
  `_to_kernel` @507, `_omega_inf_limit_fast` @460, `factor_propagator` @366,
  `_safe_factor` @422, `_run_with_timeout` @57,
  `_compute_residues_via_polynomial_fracfield` @81 — all EXACT.
- `build_G_t_matrix` @135, `_to_sr_ab` @225, `G_t_entry` @379,
  `G_t_delta_coeff` @433, `_infer_time_variable` @459 (propagator_td.py) — all
  EXACT. `_infer_time_variable` correctly named as using `is_trivial_zero()`.
- `_cofactor_adj` @1470 with the transposed index convention
  (`row j, col i`) — gotcha is correct.
- Stub `msrjd/core/propagator.py` exists, is a docstring-only stub, imported by
  nothing — gotcha holds.
- Fourier convention `fourier_transform` @field_theory.py:33 = ∫f·e^{-iωt} —
  EXACT; SymPy-first/Maxima-fallback dispatch confirmed; no `import sympy` in
  either propagator file (claim holds).
- Complexity gate `rich = nf >= 6 or len(free_syms) > 20` @688 — EXACT.
- Adjugate measurement "734 s vs ~11 s on 4×4 spike-reset" — matches code comment.
- Catastrophic-cancellation "3–5 digits / 11–42% on quad's residues",
  "10³–10⁴ divergence" — match code comments verbatim.
- Two-probe scaling test at ω=1e8 / 1e10 @1151–1152 — EXACT.
- Phase wiring: `compute.py:357` (phase 2), `:632` (phase 4); precompute
  `_precompute.py:168` at taylor_order=2 — all EXACT.
- Cache: slugify @547, staleness guards (`nf != ft._n_tilde`, spatial
  `G_tx_sym is None`) @551, save-before-attaching-G_tx @816–820,
  `make_g_tx_callables` @834 — all correct.
- Numerical-toolkit claims: `np.linalg.inv` @1069, `np.polyfit` @1095,
  `np.delete` minors @1477/1556 — all present.
- Two Newton-refine blocks @1297 and @1391 BOTH run for the rich branch
  (gate `if adj_ft is None or G_ft is None:` @1389) → "refined twice" is correct.
- OU worked example math (pole iμ, residue −i, C₀=1, G_R=Θ(t)e^{−μt}) — correct.
- Squarefree guard `return None,None,None` @207; exact-path return triple @301;
  docstring-says-`(None,None)` staleness note (docstring @111–112 vs 3-tuple
  returns) — all correct.

## Findings

- **nit** — manual cites the `_get_propagator_entry` cross-reference as
  `propagator_td.py:44`; the sentence that actually names `_get_propagator_entry`
  is at **L50** (L44 is the start of the transpose-convention block in the same
  module docstring). Right region, off by 6.
  *Location:* 07-propagator.tex:1044 (gotcha) → propagator_td.py:50.

- **nit** — manual cites the QQ-rationalise throw / fall-through gotcha as
  `pipeline/_propagator.py:122`; `_sr_complex_to_CF` is defined at **L124** and
  the throwing line `re = QQ(c.real_part())` is at **L126**. Off by 2–4.
  *Location:* 07-propagator.tex:592 (gotcha) → _propagator.py:124/126.

- **nit** — manual cites the lean-block per-entry `_omega_inf_limit_fast`
  application at `pipeline/_propagator.py:719`; the actual call
  `_omega_inf_limit_fast(entry, omega)` inside `_do_symbolic_inverse_block` is at
  **L726**. Same block, off by ~7.
  *Location:* 07-propagator.tex:884 → _propagator.py:726.

- **nit** — several other citations land a few lines before the exact target
  (all inside the correct function/block): squarefree guard "@196" → `if not
  Q_poly.is_squarefree()` @197; roots/filter "@214" → @215; residue loop "@246"
  → @247; LCM loop "@172" → @173; rich-branch sanity `free_syms` "@971"→@973 with
  WARNING print "@976"→@977; symbolic-inverse try/except SignalError "@451" (for
  `_safe_factor`) → @457. None point at the wrong file or function.
  *Location:* 07-propagator.tex (various) — informational.

- **minor (wording)** — manual §"frequency-domain propagator" states "For a
  diagonally-dominant K_ft **every** entry of G_ft shares one common denominator
  Q(ω)"; the source comment (@157–160) says "**most** entries share the same…
  denominator", and the whole point of the immediately-following LCM subsection
  is that structured (upper-triangular) K_ft entries have *different* canonical
  denominators. The "every" is locally too strong, though the chapter resolves
  the nuance correctly two subsections later (the LCM gotcha). Consider
  softening "every" → "most".
  *Location:* 07-propagator.tex:358–360 → _propagator.py:157–160.

## LaTeX validity
- All macros used (`\code \file \term \msrjd \dd \ii \ee \Dt \Gret`) and all
  environments (`note gotcha defn` tcolorboxes; `lstlisting[style=console]` style
  defined @daedalus_manual.tex:83) are preamble-defined.
- Environment balance: defn 6/6, description 3/3, enumerate 1/1, equation 9/9,
  gotcha 14/14, itemize 7/7, lstlisting 29/29, note 4/4. Brace delta 0.
  Chapters 1, sections 15, subsections 11.
- No bare `_ # & % $` in prose (a 65-hit underscore sweep was 100% false
  positives — every hit is a legitimate math subscript inside `$…$` / `\[…\]` /
  `equation`).
- No `!`-prefixed TeX errors in `daedalus_manual.log`; chapter input + typeset
  successfully.

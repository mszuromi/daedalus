# Audit: `sections/12-phasej-temporal.tex` (Phase J — Temporal Loop-Integral Engine)

**Verdict: accurate (minor-issues only).** The chapter is an exceptionally faithful
description of `final_integral.py`, `pipeline.py`, and `subgraph.py`. Every named
function/class/constant/flag, every documented return value, every control-flow
gate, and the inline code listings were verified against the live source. The few
findings below are nits (decorator-vs-def line citations) plus one cross-source
convention curiosity that is *not* a chapter error.

**LaTeX: valid (`latex_ok = true`).** All environments balanced (lstlisting 7/7,
gotcha 14/14, defn 7/7, note 4/4, equation 9/9, align 1/1, etc.). All custom macros
(`\code \file \term \ee \dd \ii \Gret \Aut \avg \msrjd`) and tcolorbox environments
(`note gotcha defn`) are defined in `daedalus_manual.tex` preamble. No bare
`_ # & % $` in text mode (the `&` live only in the `align` env; the `#` only in
`lstlisting`; `$` count even). The chapter is `\input` at `daedalus_manual.tex:200`,
and the compile log `daedalus_manual.log` (regenerated 14:44, *after* the chapter's
09:55 mtime) contains **no** Undefined-control-sequence / LaTeX Error / Misplaced /
Runaway / Missing — only cosmetic Overfull-hbox warnings. The PDF built with this
chapter present.

## Findings

- **nit — decorator-vs-`def`/`class` line citations (consistent off-by-one).**
  The chapter cites the `@dataclass`/`@njit` *decorator* line rather than the
  `class`/`def` keyword line for decorated objects. `EdgeModeSum` cited as
  `final_integral.py:116` (decorator at 116, `class` at 117); `_CausalPoset` cited
  as `:691` (decorator 691, `class` 692); `_exp_over_chain_simplex_numba_core`
  cited as `:1219` (`@_numba.njit(cache=True)` at 1219, `def` at 1220). The cited
  line is part of the construct in every case, and the convention is internally
  consistent, so these point a reader to the right place.
  Location: chapter lines 190, 627, 721 vs `final_integral.py:116/117, 691/692, 1219/1220`.

- **nit — comment-line citations land one line above the comment body.**
  Chapter says "the inline comment at `pipeline.py:291`" for the tree/loop-share-
  algorithm rationale; the comment text actually begins at line 292 (291 is the
  blank line). Same family as above; right region.
  Location: chapter line 1112 vs `pipeline.py:292`.

- **minor (NOT a chapter defect) — Fourier-kernel sign disagrees with a code
  comment, but the chapter is the self-consistent one.** Eq. (fourier-convention)
  writes `G(t)=(1/2π)∫dω e^{+iωt} Ĝ(ω)` with upper-half-plane poles and `λ=i·p`.
  The `EdgeModeSum` docstring (`final_integral.py:129-131`) instead labels the
  convention `e^{-iωt}` while *also* asserting upper-half-plane poles and `λ=i·p`.
  Those two docstring clauses are mutually inconsistent (with `e^{-iωt}` and
  Im p>0 you close downward and enclose no poles for t>0); the chapter's
  `e^{+iωt}` form is the one that actually yields the decaying `e^{ip t}` the code
  computes. The operative formula `modes_lambdas = complex(...)*1j` (`λ=i·p`,
  line 172) matches the chapter verbatim, and the chapter explicitly defers
  convention bookkeeping to the propagator chapter. No action needed on this
  chapter; if anything the stale comment is in the code.
  Location: chapter eq. line 140 vs `final_integral.py:129-131,172`.

## Spot-verified claims that are CORRECT (high-value, non-exhaustive)

- `EdgeModeSum` fields (`ri, pi, delta_coeff, modes, dt_c0, dt_int_pairs,
  dt_ext_pairs`), `frozen=True`, dt_* empty at construction — listing matches
  source exactly (`:117-147`).
- `_build_edge_mode_sums` returns `None` on missing `pole_vals`/`C_mats`; the
  `complex(CDF(SR(...)))` idiom and `modes_lambdas = ...*1j` listing match
  (`:165-200`).
- `_extract_exp_mode` log-derivative trick `λ=d/dτ log g|_0`, `C=g(0)`, returns
  `None` for `g(0)=0` (alpha kernel `τ/τ_g²·e^{-τ/τ_g}`) (`:203-232`).
- δ-subset force-smooth gate `|delta_coeff| < 1e-15`; `2**n_branch` subsets
  (`:3002,3007`).
- `_build_modesum_plan` returns exactly keys `pole_tuples`, `alphas_per_tuple`,
  `gamma_const_per_tuple`, `gamma_slope_per_tuple_per_ext` (`:664-667`); 1d/2d/nd
  modesum integrators all accept the optional `pole_tuples=` arg (`:1982,2181`).
- `_enumerate_pole_tuples` empty-edge yields `(1.0+0j, ())`, hand-rolled stack
  (no itertools.product) (`:529-564`).
- m=1 closed form `(term_U-term_L)/alpha_s`, `(U-L)·e^γ` as α_s→0, returns `None`
  on unbounded-divergent side (`:2143-2172`).
- m=2: `_exp_over_unit_triangle` J(p,q) = `((ep-eq)/(p-q)-(ep-1)/p)/q`; four Taylor
  regimes at `_J_TAYLOR_EPS=1e-6`; the doubly-degenerate moment-expansion listing
  is byte-for-byte correct (`:356-403`). Sutherland-Hodgman `_clip_polygon_to_
  halfplane` boundary-as-inside (`:454-496`); `_polygon_from_2d_constraints` bbox
  + clip listing matches (`:499-526`). `_exp_over_triangle` scales by
  `abs(det)*cmath.exp(v0_term)*J` (`:449`).
- Bilateral overflow guard `EXP_REAL_LIMIT=600` on `|Re p|,|Re q|,|Re v0_term|`;
  commit 7f0bf05 / spike-reset k=2 ℓ=1 revert narrative matches the code comment
  (`:428-446`).
- m≥3: `_CausalPoset` edges `(u,v)⇒s_v>s_u`; `_extract_causal_poset` returns
  `None` for mixed/shifted constraints (`:692-741`). `_enumerate_linear_extensions`
  recursive Kahn, deterministic order (`:806-854`). Chain simplex empty
  (`upper<=lower`)→0 with the τ<0 crossing rationale, degenerate β→None, 2^N terms
  (`:910-960`). `_exp_over_chain_simplex_polynomial` carries `(s-lower)^k`, never
  None on degenerate β (`:1339,1387`). `_chain_with_intermediate_uppers`
  effective-upper/levels/monotone-cut-tuples (`:1489-1526`). Numba core returns
  `(status,value)` with 0/1/2 = ok/degenerate-β/overflow; `cache=True` (`:1218-1226`).
  `POSET_PHYSICAL_MARGIN=50` fallback `L=earliest_ext-50`; trade-off comment at
  `:290` (`:1798-1802`).
- scipy layer: `_integrate_polytope` branches m=1/2/≥3 → `_integrate_1d/2d/nd_
  polytope` (`:4379-4408`); `_complex_quad` re/im split (`:4520-4547`);
  `_make_heaviside_filtered_integrand` returns 0 when `dt<=0` (chapter's "≤0"
  matches the *implementation* at `:4513`, not the looser docstring header);
  `_resolve_1d_bounds` degenerate `(0.0,0.0)` sentinel with the
  `quad(f,+inf,-inf)` sign-flip rationale (`:4562-4596`); deferred-constraint test
  `test_phase_J_nd_polytope_preserves_deferred_constraints` exists
  (`tests/test_time_domain.py:961`); `pure_s1_found` / ~12% k=4 overshoot
  (`:4730-4736`).
- `QUAD_OPTS={'limit':200}`, `TAU_KERNEL_CAP=50.0`, `POLYGON_BBOX_CAP=200.0`,
  `_RUNTIME_COUNTERS` keys (attempted/returned_none/scipy_nquad), `subset_
  diagnostics['evaluator']` intent field — all verified.
- Wick: `external_wick_compensation` at `symmetry.py:343`; `classify_coefficient_
  factors` exists (`symmetry.py:618`); `_all_mappings` enumeration; origin pinned
  to `SR(0)` at `cp==origin_leaf_idx` (`:2617`); per-permutation re-pinning listing
  at `:3759` matches (subtract `t_origin`), with the `V(τ)+V(-τ)` / `free_val=0`
  asymmetry narrative matching the code comment (`:3787-3791`).
- Kernels: `NoiseSourceType, ConvVertexType` imported from `msrjd.core.vertices`
  (`:90`).
- Parallelism: `multiprocessing` `fork`; `_fork_unsafe_in_notebook` is the 3-way
  AND (fork ∧ darwin ∧ ZMQInteractiveShell) via `msrjd/fork_safety.py:27-44`;
  worker entry points `_worker_eval_total_C` (`:174`), `_worker_eval_one_diagram`
  (`:186`); Windows `ValueError: cannot find context for 'fork'` (`:78`);
  `total_C_batch` (`:374`) / `eval_per_diagram_batch` (`:502`) closures inside
  `compute_correction_td` (`:202`); bit-identity tests
  `test_phase_J_total_C_batch_*` exist (`tests/test_time_domain.py:1448,1527`).
- δ-spike: `eval_delta_contributions_on_tau_grid` τ_fire=−c'/a_vary,
  coeff/|a_vary|/dτ, degenerate whole-slice (no 1/dτ) (`:3828,3932-3999`);
  `eval_delta_contributions_on_2d_grid` sweeps along larger-coefficient axis
  (`:4004,4111`).
- Tools: import `from sage.all import SR, fast_callable, CDF, solve as sage_solve`
  (`:83`); `import numpy as np` at `:1041`; mpmath `mp.dps=50` rescue paths gated by
  `USE_CHAIN_SIMPLEX_PRECISION_FIX=False`/`USE_POSET_MPMATH_ACCUMULATION=False`
  (`:1081,1723`); no networkx/sympy/nauty in the module; graph via
  `D.vertices()/D.edges()/D.num_edges()` (`:2621,2701,2408`).
- `integrate_diagram` is the real entry (def `:2411`); `integrate_tree_diagram =
  integrate_diagram` alias (`:5269`); stale "loop diagrams skipped" text in
  `pipeline.py` module header (`:16-18`) and `compute_correction_td` docstring
  (`:225`) confirmed stale — live code evaluates all loop orders. Chapter's "trust
  the code, not the header" is correct.
- `subgraph.identify_loop_subgraphs` (`:66`) returns `[]` for empty `free_freqs`,
  raises `NotImplementedError` otherwise (`:110-111` + docstring).
- Line counts: `final_integral.py` 5269 (chapter "~5,300" ✓), `pipeline.py` 573,
  `subgraph.py` 118.

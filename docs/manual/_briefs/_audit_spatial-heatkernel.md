# Audit: `sections/15-spatial-heatkernel.tex`

**Verdict: accurate** (one minor attribution imprecision, two nits). LaTeX is valid; the document compiles cleanly to PDF.

Audited against: `msrjd/integration/spatial/heat_kernel.py`, `spatial_correlator.py`, `temporal_integrate.py`, `loop_parametric.py`, plus `full_integrator.py`, `spatial_reduce.py`, and `pipeline_bridge.py` (cited but not in the four-file list).

## Summary
The chapter is unusually faithful. Every named identifier exists in the code with the described signature and behaviour. Of ~40 `file:line` citations spot-checked, all land on the correct file and within (or exactly at) the cited region. All custom environments (`note`/`gotcha`/`defn` via `\newtcolorbox`, `example` via `\newtheorem`) and all macros (`\msrjd \Lap \Gret \ee \ii \dd \code \file \term \Dt \avg \Sym`) are defined in the preamble (`daedalus_manual.tex`), `\lstdefinestyle{console}` exists, and the build log shows no LaTeX errors and the chapter rendering on pages 236–237.

## Findings

- **minor / "Its `momentum_integral` (`spatial_reduce.py:101`) evaluates [...] using `np.linalg.det` for U and `np.linalg.solve` to form Λ⁻¹N without explicitly inverting Λ."** — The `np.linalg.det` (line 92) and `np.linalg.solve` (line 96) calls are NOT in `momentum_integral` (line 101); they live one call-level deeper in the helper `symanzik_polynomials` (line 83), which `momentum_integral` calls at line 115. The math attributed to `momentum_integral` is correct and it does produce that result via the helper, so the integral description is right — only the "which function holds the `det`/`solve` calls" attribution is off by one helper. Location: chapter §sec:hk-symanzik, lines 916–923; code `spatial_reduce.py:83–97,101–122`.

- **nit / Static closed form written with `D^{noise}` numerator: `C(x,0)=D^{noise}/(2√(AB)) e^{-|x|√(A/B)}`.** — The function `free_correlator_static_closed_form` (`spatial_correlator.py:398`) writes the d=1 case as `(T / 2√(μD)) e^{−|r|√(μ/D)}`, i.e. the prefactor numerator is the parameter `T`, not a symbol literally named `D^noise`. These are the same physical quantity under the framework's noise convention (the chapter's own §sec:hk-noise establishes `D^noise` as the white-noise strength, and the OU `T`/temperature parameter is that strength; the docstring confirms `C(q,0)=T/(μ+Dq²)`). Internally consistent, not a defect — flagged only because a reader diffing the LaTeX against the source sees `T` vs `D^noise`. The d=1 exp / d=2 K₀ / d=3 Yukawa structure the chapter lists matches the code exactly. Location: chapter §sec:hk-tree, lines 668–673.

- **nit / `_use_analytic` flags cited as `pipeline_bridge.py:1667,1677`.** — `_all_plain` is at 1667 and `_use_analytic` at 1677 (both exact); the third listed line `_force_num` is actually at 1676. The citation gives the first and last line of the three-line block, so the pair is defensible. The listing's paraphrased comment (`# no derivative vertices`) differs from the source comment (`# rec=(dd,pv,ff,el,nt,ns)`) but is a correct gloss: `rec[2]` is the form-factor `ff`, and `is None` ⇒ no derivative vertex. Location: chapter §sec:hk-loops, lines 783–789; code `pipeline_bridge.py:1667,1676,1677`.

## Spot-checks that PASSED (high-value verifications)

- `gaussian_heat_kernel` (`heat_kernel.py:58`), listing matches verbatim incl. the `float(t.real)` coercion, the `t<=0 → 0j` causality guard, `(4πBt)^(-0.5*spatial_dim)` prefactor, `drift_exp` (line 89), and the d=1-only drift guard (lines 85–88). ✓
- `erf_time_integral` (`:93`), the erf-split antiderivative `F(w)`, the `w==0` special-case `_F` (lines 115–122), 30-dps `mp.workdps`, `U_hi=None ⇒ F(∞)` semi-infinite limit. ✓ (chapter cites :93 and 115–122 — exact)
- `image_sum` (`:132`), d=1-only guard (143–145), pair-summed super-exponential truncation with the `n >= 2` guard, `n_max_cap`. ✓
- `extract_mass_diffusion` (`:161`): `subs={lap_sym:-k²}`, `grad_sym→i·k`, `.expand()`, residual-operator `.has()` reject, ω-coeff `simplify_full() != 0` reject (line 201), `deg > 2` reject (line 213), `B/V/A = coefficient(k,2/1/0)`. Listing matches code lines 188–223. ✓
- `reaction_diffusion_matrices` (`:226`), returns `(M, 𝒟, V)` SR matrices, diagonal+unit-normalized iω requirement (262–268). ✓
- `build_spatial_propagator` (`:310`), all 9 returned keys (`G_tx_sym, k_var, spatial_dim, bc_mode, bc_params, initial_mode, ac_mass, ac_diffusion, ac_drift`) match lines 421–431; off-diagonal `G_tx_sym=None` (403–406); diagonality guard (369–377); periodic-d=1 guard (345–349 → message at 348); `grad_sym = SR.var('GradX')` (line 360). ✓
- `make_g_tx_callables` (`:434`), `_g` closure dispatch on `bc_mode` (image_sum vs gaussian_heat_kernel) matches listing; `_make_numeric` (`:287`) uses `expr.variables()` sorted by name, KeyError on missing param. ✓
- `extract_noise_coefficients` (`spatial_correlator.py:46`): reads `ft._by_tp[(2,0)]`, pure φ̃ᵢ² = exponent 2 on a response generator (first `n_tilde`), `D_noise = -Re(coeff)`. ✓ `extract_noise_matrix` (`:93`): `N_ii=-2·coeff`, `N_ij=-coeff`, q⁰-only. ✓
- `free_two_point` (`:138`) signature and body match; `compute_spatial_correlator_tree` (`:178`) — Tier-1 `G_tx_sym is None → NotImplementedError` (raise at 241), B<0/B=0 guard (`if B <= 0.0` at 261, message thru 275), `free_two_point(A, B, Dn, …)` at line 291. All cited lines exact/near-exact. ✓
- `free_correlator_static_closed_form` (`:398`): d=1 exp / d=2 K₀ (`scipy.special.k0`) / d=3 Yukawa. ✓
- `radial_inverse_ft` (`:299`): d=1 cosine, d=2 Hankel J₀ (`scipy.special.j0`), d=3 sin with x→0 limit, `np.trapz`, `k_max=q_grid[-1]` truncation. `periodic_inverse_ft` (`:341`): `np.meshgrid`, `np.interp`, `np.where`, `1/L^d`, cosine lattice sum. ✓ Bridge routes d≥2 periodic→`periodic_inverse_ft`, else `radial_inverse_ft` (`pipeline_bridge.py:440–441,601–602`). ✓
- `_heat_kernel_x_general` (`full_integrator.py:143`): the n_ext=1 and n_ext≥2 branches, `np.linalg.det`/`np.linalg.inv`, the two `np.einsum` patterns (`'xj,pjk,xk->px'`, `'xjc,pjk,xkc->px'`) — listing matches code lines 143–184 verbatim. ✓
- Env vars: `SPATIAL_FORCE_NUMERICAL_FT` (`pipeline_bridge.py:610,1676`), `SPATIAL_Q_CUT` (:1604), `SPATIAL_N_Q` (:1605), `SPATIAL_MEM_BUDGET_GB` default 6 (:1641, abort msg :1665). All cited lines exact. ✓
- `loop_parametric.symanzik_UF` (`:41`) U/V/W/F_reduced + `U≤0` raise; `gaussian_momentum_integral` (`:73`). `spatial_reduce.momentum_integral` (`:101`) `Q_eff = Q − NᵀΛ⁻¹N`. `loop_parametric.sigma_K_kernel` uses `dblquad` (the cited "bottleneck"). ✓
- `temporal_integrate.sigma_parametric` (`:56`): `T^{n_C}` factor (return line 123), `e^{-μΣw}` (line 92), retarded `w_e=t` (line 88), dispatch n_C=0 direct / n_C=1 `integrate.quad` / n_C≥2 Gauss–Laguerre `laggauss` + `itertools.product` (lines 94–122), substitution `s_C = lo + x_k/μ` (line 110). `bubble_edges` (:127), `sunset_edges` (:135), `bubble_delta_equal_time_via_C` (:144) reproduces `loop_dyson.bubble_delta_S` at d=1. ✓
- Oracle banner: both `loop_parametric.py` and `temporal_integrate.py` carry "⚠ ORACLE-ONLY — not on the production path"; both docstrings state "`compute_cumulants` does NOT use this module" — verbatim support for the chapter's central caveat. `grep compute_cumulants` finds it nowhere in either oracle module. ✓

## LaTeX validity
- Environments balanced: 13 equation, 7 lstlisting, 7 note, 6 gotcha, 1 each defn/example/itemize/enumerate/longtable/align*, 2 center, 4 description — every `\begin` has a matching `\end`. ✓
- No bare `_ # % & $` in prose (all such characters occur inside math mode, `longtable`/`align*` `&` separators, or escaped `\_` inside `\file{}`/`\code{}`). ✓
- All `\ref{sec:hk-*}` / `\eqref{eq:hk-*}` resolve to in-file labels; `\ref{ch:spatial-coupled}` resolves to `sections/16-spatial-coupled.tex`. ✓
- Preamble defines every macro and the `console`/`py` listing styles; build produced `daedalus_manual.pdf` with no errors in `daedalus_manual.log`. **latex_ok = true.**

# Audit: `sections/17-compute-orchestration.tex`

**Verdict:** accurate (minor-issues — only nits)
**LaTeX OK:** true

Adversarial accuracy + LaTeX audit of the chapter documenting `pipeline/compute.py`
(`compute_cumulants`). Every named function/class/file, every file:line citation, and
every behavioral claim was verified by targeted grep/Read against the live source.

## Summary

The chapter is exceptionally accurate. `pipeline/compute.py` is exactly 958 lines and
*every* cited `compute.py:NNN` line maps to the claimed construct. Cross-module
citations (`_propagator.py`, `_mean_field.py`, `_diagrams.py`, `symmetry.py`,
`time_domain/pipeline.py`, `fork_safety.py`, `type_assignment.py`, `access.py`,
`notebooks/daedalus.py`) all resolve to the right symbol at the right line. LaTeX
environments are balanced and all custom macros/environments/styles are defined in the
main `daedalus_manual.tex`. No invented symbols, no behavioral overstatements, no
gates described backwards.

## Verified (high-signal, sampled)

- Signature at `compute.py:108`, all 21 kwargs incl. defaults (`max_ell=0`,
  `parallel=True`, `spatial_parallel=True`, `spatial_n_q=64`, `mf_dae_n_starts=64`) — exact.
- Validation block `:253` (`fundamental={}`, `external_fields` required, `len==k`) — exact.
- Taylor budget `max(k+2*max_ell, 2)` at `:279`, floor-of-2 rationale — exact.
- `normalize_external_fields` (`access.py`) translation, `external_fields_user` kept,
  echoed as `external_fields_in` in config — exact (`:288`, `:902`).
- `_phase_time` closure + `phase_walls=None when not verbose` (`:296`) — exact.
- Phase[1] expand cache: `find_best_cached_order`/`prepare_for_load`/`load_expand`/
  `save_expand` from `pipeline._expand_cache`, try/except-swallow on save — exact.
- `extract_vertex_types`/`extract_source_types`/`NoiseSourceType` count — exact.
- `build_propagator` at `_propagator.py:524`; prop keys incl. `ring_gen_names` — exact
  (docstring lists exactly `K_ker,K_ft,G_ft,adj_ft,D_omega,D_delta,t_var,omega,nf,ring_gen_names`).
- `solve_mean_field` at `_mean_field.py:14`; `fsolve` import at `:11`, call at `:174`
  with `full_output=True` — exact. `mf_bg_conditions`/`phi0_i` collapse logic ~`:272` — exact.
- `solve_mean_field_dae_compat` routed on `model.get('equations')` — exact.
- Spatial branch: warn `:390`-region, k=2-needs-grid ValueError `:414`, drift-zero check
  `:450`, k≥3 `C_kpoint` early return `:463`, `parallel and spatial_parallel` `:511`,
  k=2 early return `:584`, clear-failure ValueError `:619` — all exact.
- `compute_poles_and_residues` at `_propagator.py:851`, three-tier strategy with
  `CyclotomicField(4)[ω]=QQ[i][ω]`, numpy-cofactor fallback, residue
  `C_k[i,j]=i·P_ij(ω_k)/Q'(ω_k)` — exact (docstring + `_compute_residues_via_polynomial_fracfield`).
  Mutates-prop-in-place gotcha (`:632`) — exact. No-roots warn `_propagator.py:976` — exact.
- `enumerate_unique_diagrams` at `_diagrams.py:54`; four stages `enumerate_all_typed`/
  `filter_causal`/`deduplicate_with_multiplicities` — exact. `v3` cache suffix
  `unique_typed_mult_v3_...` at `_diagrams.py:150` — exact.
- `classify_coefficient_factors` at `symmetry.py:618`, returns keys
  `Scal/scalar_prefactor/vertex_time_factors/source_time_info/is_stationary` — exact.
  `coeff = -SR(vtype.coefficient)` at `:643` — exact. `combinatorial_factor`:403,
  `_wick_leg_factor`:115, `_automorphism_order`:328, `automorphism_group(partition=…)`:339,
  `canonical_label`:517 — all exact.
- multiplicity-not-multiplied-into-prefactor (Path A), comment block `:686`-`:692` — exact.
- Phase[7] `propagator_data` dict `:714`, τ-pattern by k `:729`, zero-placeholder `:746`,
  grouped-import `:761`, `compute_correction_td` at `time_domain/pipeline.py:202` — exact.
- `total_C_batch` (pipeline.py:374) fork pool via `mp.get_context(start_method)` default
  `'fork'`, degrades through `_fork_unsafe_in_notebook` — exact; "fans τ-grid over fork
  pool or degrades to serial under the notebook fork guard" is correct.
- Master assembly `total_C` def `:802`, `C_tau` sum `:808` — exact.
- Adaptive MF dict 3-source order (naming_convention mf_parameters → mean_field flag →
  nstar/vstar/mstar), `_saddle_indices`, all-NaN drop `if not all(v != v ...)` `:878` — exact.
- Result dict keys table — every key present and correctly typed; spatial early-return
  shapes (`C_tau_x`, `C_tau_x_by_order`, `C_kpoint`) — exact.
- DAE extras block `:910`-`:944`, `linear_stability` import — exact.
- Phase-wall summary `:946`-`:955` listing — exact.
- `dd.run` (`notebooks/daedalus.py:487`) calls `compute_cumulants(**kw)` at line 586 —
  exact (chapter's "notebooks/daedalus.py:586").
- `fork_unsafe_in_notebook` at `fork_safety.py:27`, narrow darwin+ZMQ+fork gate — exact
  (chapter spells the name without leading underscore, matching the public function).
- LaTeX: env balance center/defn/equation/example/gotcha/itemize/longtable/lstlisting/
  note/enumerate all matched begin==end. Macros `\code \file \term \msrjd \Daedalus \Sym
  \Aut \ee \ii \dd`, envs `gotcha defn note example`, style `console` all defined in
  `daedalus_manual.tex`; chapter is `\input`/included. No bare `_ # % & $` outside
  listings/math.

## Findings

- **nit** — manual claim: "The docstring (`pipeline/compute.py:179`) records that this
  floor was *historically 4*" (chapter §co-args, line ~265). — code reality: the
  `**Previously this floor was 4**` docstring sentence is at line **180**, not 179
  (line 179 is the blank line before it; the surrounding taylor_order docstring paragraph
  spans ~165–184). Correct file, correct paragraph, off by one line. — location:
  17-compute-orchestration.tex:265 → compute.py:180.

- **nit** — manual claim: "the code warns (`pipeline/compute.py:390`) and falls through
  to the temporal path." — code reality: the `if spatial_grid is not None and not
  model.get('spatial'):` guard is at line 389 and the `_warnings.warn(...)` call body
  spans 391–394; line 390 is `import warnings as _warnings`. The cite points at the right
  3-line block. — location: 17-compute-orchestration.tex:531 → compute.py:389–394.

- **nit** — manual claim: "checks that the drift $V$ is zero
  (`pipeline/compute.py:450`; nonzero drift raises)." — code reality: line 450 is the
  `if not _np.allclose(_Vn, 0.0, atol=1e-9):` check; the `raise NotImplementedError`
  itself is at line 451. The cite labels the check line, and the raise is the next line —
  the parenthetical "nonzero drift raises" is accurate to the block. — location:
  17-compute-orchestration.tex:572 → compute.py:450–452.

No critical, major, or minor findings. The three nits are off-by-one/region line cites
that all point at the correct file and correct logical block.

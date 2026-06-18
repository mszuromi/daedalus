# Cross-Chapter Consistency Audit — Daedalus Manual

Lens: do any two chapters contradict each other or drift in notation/terminology?
Method: traced each shared concept the brief names across all `sections/*.tex`,
read the load-bearing passages, compared formulas/symbols/flag-names verbatim.

Files audited (relevant passages): 02-quickstart, 03-theory-spec, 04-theory-files,
07-propagator, 08-mean-field, 09-enumeration, 10-type-assignment, 11-caching,
12-phasej-temporal, 13-grouped-phasej, 14-spatial-core, 15-spatial-heatkernel,
16-spatial-coupled, 17-compute-orchestration, 18-engine-api, A-glossary, C-file-index.

---

## VERDICT BY CONCEPT

### 1. Symmetry factor S(Gamma) vs M(Gamma) — CLEAN (no contradiction)
- Written `\Sym(\Gamma)` **everywhere** (ch10, ch11, ch17, ch14, ch15, glossary,
  B-external-tools, quickstart). No stray `M(\Gamma)` as the symmetry factor
  (`grep -F 'M(\Gamma)'` → zero hits).
- `M_v` appears ONLY in ch10 (note around line 860-869) as the explicitly
  *deprecated* per-vertex "Path B" alternative, with the text stating production
  uses the Aut-based `\Sym(\Gamma)` (Eq. ta-sym), not `M_v`. This is correct and
  self-consistent.
- Formula agrees across ch10 (eq:ta-sym), ch11 (~l.892), ch17 (eq:co-Sym),
  glossary (boxed defn ~l.1353): `prod_v prod_l n!/|Aut_fixed-ext|`. The
  "Path A / multiplicity is diagnostic-only" message is consistent in all four.
- NIT: numerator written two ways — ch10/ch17 `\prod_v \prod_\ell n_{v,\ell}!`
  vs ch11/glossary `\prod_{legs} n_leg!`. Same object, harmless abbreviation.
- NIT: `\Aut` subscript hyphenation drifts — ch10 uses "fixed ext" (9×, no hyphen);
  ch11/ch17/glossary/C-file-index use "fixed-ext" (hyphen). Cosmetic.

### 2. Propagator G_ft, poles, residues — CLEAN on naming, ONE SIGN BUG (see MAJOR)
- `G_ft` (code) / `G_{\mathrm{ft}}` (math) for frequency-domain propagator;
  `G_R(t)` / `\Gret` time-domain retarded; `G^R` response convention;
  `D_delta`/`D_\delta` instantaneous piece; residue matrix
  `C_mats[k] = i * Res G_ft` (the "C_k convention"). All consistent across
  ch07, ch12, ch13, ch17, ch08, glossary.
- Retarded-pole half-plane `Im(omega) > 0` is uniform everywhere
  (ch07, ch08, ch10, ch17, B-external-tools, C-file-index, glossary).
- BUT the time-domain reconstruction exponential sign is contradicted in two
  spots — see MAJOR finding below.

### 3. Phase-J analytic-vs-quad — CLEAN
- ch12 and ch13 agree: dedicated analytic closed-form integrators for m=1,2,>=3,
  with a scipy.quad / scipy.nquad fallback on degeneracy / FP overflow / unbounded
  endpoints (evaluator returns None -> scipy).
- Gate flag names match: `USE_POLYGON_M2_INTEGRATOR` (m=2), `USE_POSET_INTEGRATOR`
  / "causal-poset integrator" (m>=3), `USE_1D_INTEGRATOR` (m=1),
  `USE_NUMBA_CHAIN_SIMPLEX`, `USE_GROUPED_ANALYTIC_MODESUM` (grouped).
- "Phase J" named consistently (`Phase~J`), aligned with the `[7/7]` / `[7]`
  pipeline step in ch02 and ch17.

### 4. Cache keys — CLEAN
- `PipelineCache._stage_key = (stage, k, loop_order)` -> filename stem
  `stage_kN_lM` (ch11 l.244). Glossary (l.1048) says keyed by `(stage, k, ell)`
  — consistent (loop_order == ell; the manual uniformly maps code `l`/`loop_order`
  to math `\ell`).
- Two-cache split is consistent: **expand cache** keyed by `taylor_order` only;
  **enumeration cache** keyed by finer tuple `(model, taylor_order, k, ell,
  external_fields)` (ch11 l.114-122). ch04's sample manifest entry
  `unique_typed_mult_v3_dx1_dx1_taylor6_k2_l2` with `"k":2,"loop_order":2`
  matches both ch11's stem format and the enumeration-cache tuple exactly.

### 5. Config `parameters` vs deprecated `fundamental` — CLEAN (well-disambiguated)
  Three distinct things are kept distinct correctly:
  - `DEFAULT_FUNDAMENTAL` — module constant in `.theory.py` (NOT deprecated).
  - `Config.parameters` — canonical Config field.
  - `Config.fundamental` — the *deprecated alias* of `Config.parameters`
    (ch18 l.446, l.483-493: `parameters` canonical, `parameters` wins if both set,
    `__post_init__` mirrors them). ch02 l.317-321 says the same.
  - `fundamental` as the `compute_cumulants(...)`/`solve_mean_field(...)` argument
    — the engine-internal dict name (ch17, ch18 l.242 "the engine's name"),
    legitimately distinct from the user-facing `Config` field.
  No chapter calls `parameters` deprecated or presents `Config.fundamental` as
  primary. Consistent.
  - Minor doc-altitude note (not a contradiction): ch17 documents the low-level
    `compute_cumulants` with `fundamental` as a positional arg while ch02/ch18
    steer users to `Config.parameters`; correct because they describe different
    layers, but a one-line cross-link would help.

### 6. k / max_ell / loop-order — CLEAN, one symbol-mix NIT
- `k` = external-leg/cumulant order; `\ell` = loop order; `max_ell` = Config/compute
  cutoff arg; `loop_order` = cache/internal name = `\ell`. Consistent.
- NIT: loop order is written `$\ell$` almost everywhere but `$L$` in ch02 l.309
  ("The loop order $L$") and ch09 l.444 ("A tree has $L=0$"). Even within ch02,
  l.309 uses `$L$` and l.454 uses `$\ell$`. Two glyphs for one quantity; each is
  defined locally so not a contradiction, but mild drift.

### 7. Spatial Symanzik U/F — NOTATION DRIFT (see MAJOR/MINOR)
- ch14 uses macros `\Usym` (U) and `\Fsym` (F) for first/second Symanzik;
  ch15 uses plain `U` and `F_{reduced}`; ch16 uses neither macro (relies on
  `\mathcal{B}`). Same first Symanzik `det Lambda` typeset two ways (calligraphic
  vs plain). More substantively, "second Symanzik" / the letter F names two
  different objects across chapters — see MAJOR below.
- `Q_eff = Q - N^T Lambda^{-1} N` (Schur complement) is consistent across ch14,
  ch15, glossary.
- `\mathcal{B} = D * Q_eff` (heat-kernel width) is consistent across ch14, ch15, ch16.

### 8. Other shared concepts spot-checked — CLEAN
- Response field `\tilde\varphi` (tilde), code prefix `t`/`d` — uniform
  (ch03, ch04, ch07, glossary).
- Fourier convention forward `e^{-i w t}` — single fixed convention, stated
  authoritatively in ch07 (eq:ftconv) and ch12 (l.140 inverse). 
- Pipeline phase numbering `[1]..[7]` (expand, propagator, mean_field, poles,
  diagrams, classify, phase_j) identical in ch02 trace and ch17 table.

---

## FINDINGS (ranked)

### MAJOR — time-domain propagator exponential sign contradicts itself across chapters
The pipeline-wide Fourier convention is fixed: forward transform `F(w)=∫f(t)e^{-iwt}dt`
(ch07 eq:ftconv), so the INVERSE carries `e^{+iwt}` (ch12 l.140 writes it explicitly
`G(t)=1/2π ∫dw e^{+iwt} Ĝ(w)`), poles sit at `Im(w)>0`, and the reconstruction
`Σ_k C_k e^{+i w_k t}` decays for t>0. This is stated and used correctly in:
- ch07 l.929 `Θ(t)Σ_k C_k e^{+i ω_k t}`
- ch12 l.147,154 `e^{+i p_α t}` (decays for t>0)
- ch13 l.160,167 `C_k e^{+i p_k t}` (decays for t>0)
- glossary l.1176 (Residue-matrix entry), l.1185 (Retarded-propagator entry):
  both `Σ_k C_k e^{+i ω_k t}`.

But TWO passages write the **opposite** sign `e^{-i ω_k t}`:
- **ch17 eq:co-Gt (l.425):** `G(t) = Σ_k C_k e^{-i ω_k t} Θ(t) + D_delta δ(t)`.
  This is *internally* inconsistent too: the sentence immediately above (l.419-421)
  says the upper-half-plane poles give "a sum of **decaying** exponentials," yet
  `e^{-iω_k t}` with `Im(ω_k)>0` GROWS for t>0. So it is a typo, not an alternate
  convention.
- **A-glossary, Pole / pole-residue-form entry (l.1055):** `Σ_k C_k e^{-i ω_k t}
  + D_delta δ(t)` — directly contradicts the glossary's own Residue-matrix (l.1176)
  and Retarded-propagator (l.1185) entries two items later.
Impact: a reader cross-referencing the time-domain propagator between the
orchestration chapter / glossary and ch07/12/13 sees a sign flip; taken literally
the ch17/glossary form is a growing (acausal) exponential. FIX: change both
`e^{-i ω_k t}` to `e^{+i ω_k t}` to match ch07 eq:ftconv and ch12 l.140.

### MINOR — "second Symanzik polynomial" / letter F names two different objects
- ch14 (eq:UF l.355, l.374-381) defines `\Fsym ≡ \Usym · Q_eff` (the homogeneous
  second Symanzik) and reserves the name "Schur complement" for `Q_eff` itself.
- ch15 (l.894-899) introduces `F_{reduced} = W - V²/U`, which equals `Q_eff` (the
  Schur complement), and calls IT a Symanzik form — i.e. ch15's "F" = `Q_eff`,
  which differs from ch14's `\Fsym = \Usym·Q_eff` by a factor of `\Usym`.
- glossary (l.1143, l.1225) calls `Q_eff` "**the second Symanzik polynomial** in
  its Schur-complement form" — siding with ch15, contradicting ch14's usage where
  the second Symanzik is `\Usym·Q_eff` and `Q_eff` is "the Schur complement."
Impact: the term "second Symanzik polynomial" and the glyph "F" are attached to
two objects differing by `\Usym`. A reader importing "F" from ch14 into ch15's
formulas (or vice-versa) would be off by `det Λ`. The MEMORY-noted tropical work
uses `F^{tr}` for the genuine second Symanzik (= `\Usym·Q_eff`), so ch14's usage
is the standard one. FIX: in ch15/glossary, either rename `F_{reduced}` to avoid
"second Symanzik" (call it the Schur complement / reduced external form `Q_eff`),
or add one sentence flagging `\Fsym = \Usym·Q_eff` vs the reduced `Q_eff`.

### MINOR — first Symanzik typeset `\Usym` (𝒰) in ch14 but plain `U` in ch15
Same object `det Λ`; ch14 uses the calligraphic macro, ch15 uses plain `U`
(l.896 `U=Σ a² w`, l.908 `computes U,V,W`). Cross-chapter glyph drift for one
quantity. FIX: standardize on `\Usym` in ch15's prose math, or note the
equivalence once.

### NIT — loop order written `$\ell$` vs `$L$`
`$L$` used for loop order in ch02 l.309 and ch09 l.444 while `$\ell$` is the norm
(and ch02 l.454 itself uses `$\ell$`). Pick one symbol.

### NIT — `\Aut` subscript "fixed ext" vs "fixed-ext"
ch10 spells it "fixed ext" (no hyphen, 9×); ch11/ch17/glossary/C-file-index use
"fixed-ext". Cosmetic; standardize the hyphen.

### NIT — symmetry-factor numerator abbreviation differs
ch10/ch17 `\prod_v \prod_\ell n_{v,\ell}!` vs ch11/glossary `\prod_{legs} n_leg!`.
Same meaning; harmonize if desired.

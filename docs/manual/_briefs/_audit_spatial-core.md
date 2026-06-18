# Audit — `sections/14-spatial-core.tex` (The Spatial Integrator: Symanzik Polynomials and Causal Chambers)

**Verdict: accurate (minor-issues).** Every named identifier, class, dataclass field,
function signature, default argument, control-flow claim, and line citation in the chapter
was checked against the live source and verified. The LaTeX compiles cleanly (the master
`daedalus_manual.pdf` was rebuilt today with the chapter `\input` at line 207; the log shows
zero fatal `!` errors for this chapter, only cosmetic Overfull-hbox warnings). All
environments balance exactly (lstlisting 16/16, equation 10/10, gotcha 9/9, align 2/2, etc.),
and there are no unescaped `_ # % & $` inside `\file{}`/`\code{}`/text — verified
programmatically and confirmed by the successful build.

The findings below are all **minor / nit**. None is a correctness or compile problem.

---

## Findings

- **minor — fabricated source quote "dead at the saddle".**
  Manual claim (§"Assembling a correlator", around the `correlator_2pt` listing): "skipping
  diagrams whose prefactor is numerically dead (`|pre| < 10^{-14}`, ``dead at the saddle'')"
  — the phrase is rendered in LaTeX double-quotes, which reads as a verbatim source quote.
  Code reality: `grep -rn "dead at the saddle"` across the whole `spatial/` package returns
  **nothing**; the phrase exists nowhere in the codebase. The *behaviour* is correct
  (`correlator_2pt` body, `full_integrator.py:1191`, does `if abs(float(pre)) < 1e-14: continue`),
  but the quoted phrase is invented.
  Location: chapter §"Assembling a correlator" (the `correlator_2pt` paragraph, ~line 854);
  code `full_integrator.py:1182-1195`.

- **minor — the time-window upper bound is overstated as `W = 22/μ`.**
  Manual claim (§"The kinematic integral", step 2, citing `full_integrator.py:495-499`):
  "$[\text{lo},\text{hi}]$ comes from the external times and a window $W=22/\mu$ --- about
  twenty-two relaxation times, well past where the integrand has decayed." This reads as if
  **both** bounds use the 22/μ window.
  Code reality (`full_integrator.py:496,499`): `W = 22.0/mu` is used **only** for the lower
  bound, `lo = mn - W`. The upper bound is `hi = me + 3.0/mu` — three relaxation times, not
  twenty-two. So "about twenty-two relaxation times" describes the past-side extent only; the
  future-side extent is 3/μ.
  Location: chapter §"The kinematic integral", step 2 (~line 495–499 of the chapter);
  code `full_integrator.py:495-499`.

- **nit — MC-bias comment line citation slightly tight.**
  Manual claim (§"The three backends", the Monte-Carlo gotcha): "This is stated repeatedly in
  the source (`full_integrator.py:560-563`)." The relevant comment block actually spans
  `558-563` (the sentence "Validated for PLAIN vertices; derivative-vertex form factors are
  biased — det Lam→0 singularity" is at lines 562-563). The cited range 560-563 lands inside
  the block, so a reader finds it; just not the first line of the comment.
  Location: chapter §"The three backends" (MC gotcha, ~line 820); code `full_integrator.py:558-563`.

---

## Spot-checks that PASSED (representative, not exhaustive)

Identifiers / files — all exist with the cited roles:
- `full_integrator.py`: banner line 4 ("ONE genuine integral evaluates *every* enumerated
  diagram") ✓; `_momentum_factor_batch`@56 ✓; `_symanzik_kernel_batch`@94 ✓;
  `_heat_kernel_x_general`@143 ✓; `_gl_on`@349 ✓ (body matches the listing: `t = upper -
  span*v*v`, `w = span*(wg*v)`); `_diagram_bessel_xs`@362 ✓; `diagram_kinematic`@469 ✓
  (full signature `(descr,q_vec,external_times,mu,D,spatial_dim=1,W=None,n_t=22,n_s=24,
  formfactor=None,gh_order=6,xs=None,method='grid',mc_n=1000000,mc_seed=0)` byte-identical);
  `diagram_value`@1142 ✓; `_is_retarded_type`@1156 ✓ (returns `kinds == ['C','R']`);
  `diagram_correlator`@1166 ✓; `correlator_2pt`@1182 ✓; `diagram_value_x`@1203,
  `diagram_correlator_x`@1213, `correlator_2pt_x`@1228 ✓.
- `2^{-n_C}` applied in `diagram_value`/`diagram_value_x` at lines **1153** and **1210** — both
  citations exact ✓.
- `momentum_routing.py`: `RoutingResult`@50 ✓ (all 4 fields match); `edge_k2`@57 ✓;
  `edge_coeffs`@62 ✓; `_num`@79 ✓; `route_momenta`@99 ✓; residual convention prose @102-108
  ✓; the conservation-loop listing (`if w==v: s+=kk[e]` / `if u==v: s-=kk[e]` / leaf
  `s-=q[...]`) matches code 120-130 ✓; `sp.linsolve` + inconsistency raise @138-142 ✓;
  free-symbol relabel @149-158 ✓; tree/bubble docstring @29-34 ✓; `edge_k2` warning in
  docstring @72-74 ✓.
- `spatial_reduce.py`: `I_mom` @12 ✓; Λ/N/Q matrix defs @18-21 ✓; collapse formula @24-26 ✓;
  validated bubble (`U=w1+w2`, `Q_eff=w1w2/(w1+w2)`) / sunset (`U=w1w2+w2w3+w3w1`,
  `Q_eff=w1w2w3/U`) @36-38 ✓; `symanzik_matrices`@58 (`aw=a*w[:,None]; Lam=aw.T@a;
  N=aw.T@b; Q=(b*w[:,None]).T@b` byte-identical) ✓; `symanzik_polynomials`@82 (uses
  `np.linalg.solve`, not `inv`) ✓; `momentum_integral`@101 ✓.
- `causal_chambers.py`: the "one difference / close-pair cannot arise" comment @17-24 ✓ (quoted
  text matches); `causal_chambers`@41 ✓ (body `_CausalPoset(...)` +
  `_enumerate_linear_extensions` matches the listing); empty-edge ⇒ all n! orderings, in the
  docstring ✓; reference integrator `integrate_over_chambers`@84 ✓ (sums `integrate_chamber`,
  which does nested `scipy.integrate.quad`).  [Chapter does not separately name
  `integrate_chamber`; it attributes the nested-quad behaviour to `integrate_over_chambers`,
  which is fair since the latter calls the former.]
- `diagram_descriptor.py`: `CEdge`/`CStackDiagram` frozen dataclasses with the exact fields
  listed (incl. `fpairs: tuple = ()`) ✓; "no bubble/tadpole branch" comment @10-13 ✓;
  `is_tadpole_like`@93 (docstring "Diagnostic only; the evaluator does not branch on it") ✓;
  `diagram_to_cstack`@105 ✓; the 7-step mapping (VertexType/2-point-SourceType/≥3-leg-source/
  `NotImplementedError` for <2 legs; 2-point-noise contraction with degree≠2 raise;
  remaining G_R → R; external flag) all match code 105-193 ✓; "C edge may take either half's
  routing" @168 (`a, b = ec[inc[0]]`) ✓.
- `generic_evaluator.py`: ORACLE banner @2-4 ✓ (quoted text matches); branches on
  `is_tadpole_like()` in `diagram_delta_C` (line 311, inside def @303, splitting
  `tadpole_delta_C` vs `bubble_delta_C`) ✓; Dyson-convolution route (`_dyson_retarded`/
  `_dyson_keldysh`, self-energy `σ_Γ(q,u)`) ✓.

Batched-core gotchas:
- `ok = U > u_floor` with `u_floor=1e-300` ✓ (lines 56,81); heat-kernel width filters
  `Bcal_k > 1e-300` / `det(Bcal_k) > 1e-300` ✓ (lines 629-630). Degenerate ⇒ silent zero ✓.
- The `_momentum_factor_batch` listing (Q/Lam/N einsums, `U=np.linalg.det(Lam)`, `ok=U>u_floor`,
  `LamiN=np.linalg.solve`, Qeff Schur complement, pref `(4πD)^{-Ld/2}U^{-d/2}`) matches code
  73-90 line-for-line ✓.

σ-grid gotcha (the subtle one) — fully correct:
- `s_cap = 32.0/mu` ✓ (line 521); `s = s_cap·v²`, `s_w = wv·s_cap·vv·exp(-mu·s_nodes)` ✓
  (lines 524-525); it IS Gauss–Legendre+substitution on `[0,s_cap]`, not Gauss–Laguerre ✓;
  the resolve-self-loop-`σ^{-d/2}`-spike comment is at line 519 ✓; **and** the chapter's
  caveat that the function docstring at `full_integrator.py:482` still mislabels this
  "Gauss–**Laguerre**" is correct — line 482 docstring literally says "by Gauss–**Laguerre**"
  while the body runs Legendre. Good catch, accurately reported.

diagram_kinematic step-by-step (chapter §"The kinematic integral") — every cited region checks:
- structure extract 487-493 ✓; poset/`s_up`/`s_lo` 503-515 ✓; Schwinger grid 517-536 ✓;
  method dispatch (`'bessel'`→`_diagram_bessel_xs`, else `causal_chambers`) 544-553 ✓;
  MC nested Exp(μ) 564-578 ✓; grid `_gl_on` 586-603 ✓; edge-weights/`mu_resid`/`amp`
  606-623 ✓ (R: `tv-tu`; C: `|tu-tv|+σ`; mu_resid excludes σ — matches "mass damping" prose);
  four accumulate sub-cases 625-709 ✓; return real-for-IFT / complex-for-derivative / float
  714-716 ✓.

Memory-wall gotcha (attributed to the bridge) — verified in `pipeline_bridge.py`:
- `SpatialPropagatorError` imported from `heat_kernel` (bridge line 52) and raised when
  `_peak_gb > _budget_gb` (1650-1653) ✓; env knobs `SPATIAL_INTEGRATOR` (1621),
  `SPATIAL_MEM_BUDGET_GB` (1641), `SPATIAL_GRID_NT`/`NS` (1365-1366) all exist ✓; the
  escape-hatch list (lower max_ell / =mc / =bessel / coarsen NT,NS / raise MEM_BUDGET_GB)
  matches 1658-1665 ✓. The specific number "KPZ 2-loop n_V=4,n_C=3 ≈ 1.8×10^8 pts/chamber,
  tens of GB" matches the code comment at bridge line 1637-1638 verbatim ✓.

Bridge data-flow (§"What feeds and consumes this subsystem"):
- `build_pipeline_records`@208 ✓; `pv = float(SR(pre).subs(...))` saddle eval (1193) ✓;
  `|pv| < 1e-14` drop (1196) ✓; `dd = diagram_to_cstack(td)` ✓; the cited
  `diagram_correlator_x(dd, pv, xg, float(tau), mu0, D0, spatial_dim=d, n_t=nt, n_s=ns,
  formfactor=ff, ...)` call is at **pipeline_bridge.py:1697** — exact ✓; output array shape
  `(n_τ, n_x)` per loop order ✓.

LaTeX:
- All custom macros used (`\code \file \term \msrjd \dd \ii \ee \Lap \avg \Gret \Sym \Usym
  \Fsym`) and environments (`note gotcha defn`, listing styles `py`/`console`) are defined in
  `daedalus_manual.tex` preamble ✓.
- Environment begin/end perfectly balanced; lstlisting 16/16; no unescaped specials in
  `\file{}`/`\code{}`; PDF rebuilt today (Jun 18 14:44) with no fatal errors → **latex_ok = true**.

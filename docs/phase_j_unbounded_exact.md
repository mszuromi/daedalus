# Phase J without an integration box: the exact unbounded routes

Status: these are the only Phase J routes. The switches that selected the older boxed routes
(`DAEDALUS_PHASE_J_DD_KERNELS`, `..._SEMIINF_CHAIN`, `..._UNBOUNDED_2D` and the umbrella
`DAEDALUS_PHASE_J_LEGACY`) are retired: setting one warns and has no effect. The box-free routes were
the default from 0.2.0. Code: `engine/integration/time_domain/boxfree.py` (kernels),
`final_integral.py` (routing, certificate, divergence policy), `expdd.py` (divided differences),
`dbm_integral.py` (the exact difference-constraint route). The public description is the 0.2.0 entry of
`CHANGELOG.md`.

## 1. Setting

After δ-elimination and the pole expansion, every Phase J subset is

    I = Σ_t C_t ∫_R exp(α_t · s + γ_t) ds,      R = { s ∈ R^m : a_e · s + c_e > 0 for every row e }

with `C_t` the product of residues of a pole tuple `t`, `α_t = Σ_e λ_{t,e} a_e`, `γ_t =
Σ_e λ_{t,e} c_e` (the external times enter `c_e`), and one row per smooth propagator edge
(`Δt_e > 0`). Before 0.2.0 the open directions of `R` were closed by a box: the chains of three
or more times at `earliest external time − 50` (`POSET_PHYSICAL_MARGIN`), the two-time polygon
at ±200 (`POLYGON_BBOX_CAP`), the DBM at ±200.

## 2. The theorem and its scope

**Scope.** Every row is a difference row of an edge, `s_v − s_u + c > 0` (or `s_v + c > 0`
against an external time), with unit coefficients; every mode of an edge decays (`Re λ < 0`).
A ConvVertex kernel is an edge (its attached leg has its own time `s`, the kernel edge
`s → anchor` has the row `anchor − s > 0` and one decaying mode). Rows that are not edges
(the NoiseSource `τ_v` box rows) keep their box (`TAU_KERNEL_CAP`), and such subsets go to
the DBM / the hardened fallback.

**Statement.** On such a region the integral of each exponential converges iff the exponent
decays along every open direction of `R`:

* m = 1, `L < s < U`: an infinite side needs `Re α` of the decaying sign;
* m = 2: `Re(α · r) < 0` for every extreme ray `r` of the recession cone of `R`, and `R`
  contains no line;
* m ≥ 3, a chain `L < s_1 < … < s_m < U` of a linear extension of the causal poset: with
  `L = −∞` every prefix sum `P_j = α_1 + … + α_j` has `Re P_j > 0`; with `U = +∞` every
  suffix sum has `Re < 0`.

and then equals the closed forms of §3. For a product of decaying propagators on edge rows
every pole tuple satisfies it (the exponent along an open direction is a positive combination of
edge rates), so failures come only from non-decaying modes (a zero-rate mode, e.g. the massless
q = 0 mode of a spatial model) or non-edge rows — the convergence certificate (M8) checks it per
tuple with a margin of 64 rounding units of the exponents (`_decay_tol`), so that the rounding
residue of an exactly constant direction never passes as decay.

**Exactly cancelling groups.** Tuples sharing `α` form a group; when the group's merged
coefficient `Σ C e^{γ − γ_max}` is exactly 0 in floating point the integrand of the group is
identically zero, and a failed certificate is not a divergence: the group is dropped (counter
`divergent_cancelled`; the chain route merges groups before evaluating chains). Anything else is
a divergent subset, handled by the divergence policy (`on_divergence` 'fallback' (default) /
'warn' / 'raise'; the spatial certification passes 'fallback').

## 3. Formulas (all in the log domain, γ folded before any exp)

* Interval: `∫_L^U e^{αs+γ} = (U − L) exp[αL+γ, αU+γ]` (first divided difference anchored at
  the larger real exponent); `L = −∞`: `e^{αU+γ}/α`; `U = +∞`: `−e^{αL+γ}/α`.
* Polygon: `R` is built exactly (rational arithmetic on the rows' floats): finite vertices and
  the extreme rays of the recession cone, ordered by angle into a convex projective polygon,
  fanned from a finite vertex `x0` into triangles `|det| exp[f(x0), f(u), f(w)]`, half-strips
  `|det(u − x0, r)| exp[f(x0), f(u)] / (−α·r)` and wedges
  `|det(r1, r2)| e^{f(x0)} / ((−α·r1)(−α·r2))`, `f(x) = α·x + γ`.
* Chain, `L = −∞`: `e^{S U} / Π_j P_j` (semi-infinite chain, from the dominant eigenvector);
  with intermediate upper bounds, the transfer matrix with projections (`expdd`); `U = +∞`:
  the mirror image.
* The overflow test is made once, on the final exponent of the subset (or of the folded chain
  term): a value beyond e^700 is `OVERFLOW`, never `inf`/`nan`.
* Many-tuple regions are evaluated by compiled batched kernels (M7b) that answer only the plain
  case and otherwise hand back to the scalar loop.

A finite lower bound `PHASE_J_T_MIN` (switch-on transients) is honoured by every box-free route;
a region that would reach the `scipy` fallback with it raises (the fallback cannot honour it).

## 4. The Θ(0) convention and the tie policy

* A constant row left by δ-elimination (an edge whose `Δt` is constant) is decided by the Itô
  rule Θ(0) = 0: value > 0 keeps the region, < 0 empties it, exactly 0 empties it — the same
  convention that evaluates the k = 2 equal-time point as the left limit τ → 0⁻
  (`_const_row_verdict`).
* A row comparing two coincident external times follows the external-time tie order
  (`_tie_order_sign`), so k ≥ 3 values at coincident times are one-sided limits.
* The pre-0.2.0 rule (`'legacy_clip'`, Θ(0) = 1 in the two-time polygon, with the boxed routes) is
  retired together with its switch `DAEDALUS_PHASE_J_LEGACY` (set, it warns and is ignored); the
  pre-0.2.0 numbers are kept as frozen baselines
  (`tests/fixtures/phase_j_legacy_baseline.npz`, `tests/phase_j_refactor_fixtures/legacy/`).

## 5. Post-mortem: the trap

* What happened (plan §2): the two-time polygon clipped half-planes with `f ≥ 0`, so a
  zero-normal row with `c = 0` kept the whole polygon (Θ(0) = 1) while every other integrator
  used Θ(0) = 0. Shipped values of δ-containing models were wrong (spike reset k = 2 ℓ = 1 up to
  285 % on the frozen fixture, a sign flip at τ = 3).
* Why it hid: the bilateral |Re| > 600 guard of the legacy triangle sent the affected regions to
  `scipy.nquad`, whose box happened to mask the inconsistency at the default cap; at other caps
  (cap 12) the trap value appeared. An earlier "wrong-direction" correction (7f0bf05) was this
  bug, masked by the guard (G1 replay, plan §14).
* A second defect hid behind the same guard: the legacy closed form multiplied
  `exp(γ) · exp(v0) · J` separately, so `exp(v0)` underflowed where a large γ would have
  compensated. Folding γ into the anchored divided differences (F5) removed it.
* Lessons kept: one shared Θ(0) helper for every integrator; guards are not correctness
  arguments (relax them only with the DD kernels); every route compared per region against
  independent references (tight quadrature, the 50-digit fan, the DBM) on the final tree
  (G1 condition 2 at M9: 0 disagreements except references whose own box truncates a slow mode);
  frozen fixtures refrozen once, only against an independent physical check (the ħ-extrapolated
  simulation, χ² 5.8/10).

## 6. Guards and comments in the code that refer to this

`_decay_tol` (the certificate margin), `_divergence_verdict` (the policy), `_DD_BATCH_MIN`
(where the batched kernels pay), `_dd_accuracy` (the subset-sum error accounting, counters
`dd_err_ratio_max` / `dd_ill_conditioned`). (The legacy triangle with its bilateral guard and the
`DAEDALUS_PHASE_J_LEGACY` umbrella are retired.)

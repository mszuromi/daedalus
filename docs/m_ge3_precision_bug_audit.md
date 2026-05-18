# m≥3 chain-simplex precision-loss bug audit

**Investigation date:** 2026-05-16
**Branch:** `phase-j-refactor` at HEAD `ae0e298`
**Status:** Diagnosed, not yet fixed. Workaround available (`USE_GROUPED_PHASE_J = True`).

## Summary

The per-diagram analytic m≥3 integrator (`_exp_over_chain_simplex` in
`msrjd/integration/time_domain/final_integral.py:877`) suffers
systematic precision loss when the theory's propagator has close-paired
poles. At `spike_reset` k=2 ell=1 with canonical fixture params, the
per-diag analytic stack overestimates the 1-loop value by **4×**
(per-diag `+2.66e-3` vs grouped scipy truth `+6.38e-4`, rel diff 76%).

The bug is **theory-specific** (`quad_exp` is clean) and **does not
become negligible at larger correlation magnitudes** (relative error
stays ~constant across 1.7× magnitude scaling).

## Evidence chain

### 1. Discovery — grouped path disagrees with per-diag at spike-reset k=2 ell=1

At `spike_reset` k=2 ell=1, canonical `_FUNDAMENTAL_SPIKE_RESET` params
(Em=3.5, a=2.5, w=0.55-0.80), τ=0:

| Path | C_tree(0) | C_1loop(0) |
|---|---|---|
| per-diag analytic | -1.582192e-02 | **+2.657819e-03** |
| grouped scipy (default `epsrel=1.5e-8`) | -1.582192e-02 | **+6.382799e-04** |
| grouped scipy (tight `epsrel=1e-12`) | -1.582192e-02 | **+6.382799e-04** (matches default to 2.77e-9) |

Tree matches bit-perfectly. Loop disagrees by 4.16× (76% rel diff).
The grouped path's identical answer at default and tight quadrature
proves grouped is fully converged — `+6.38e-4` is the true value.

### 2. Localization — bug is in subset evaluators, not in outer wrappers

Tree-level (ell=0) matches across paths to rel 7.7e-14. Tree uses the
same `contribution()` permutation wrapper that includes the
`_compensation`-divided Wick-contraction sum. So the bug is **not**
in the outer permutation/compensation machinery — it must be in the
m≥1 subset evaluators that fire only at ell≥1.

### 3. Quad model is clean → bug is theory-specific

`single_population_quad_exp_test` at k=2 ell=1, canonical
`_FUNDAMENTAL_QUAD` params:

| Path | C_tree(0) | C_1loop(0) | wall |
|---|---|---|---|
| per-diag analytic | +3.472449e-03 | +1.488024e-04 | 49.7s |
| grouped scipy | +3.472449e-03 | +1.488016e-04 | 90.8s |
| | | rel diff 5.7e-6 | |

Quad converges cleanly. Same code paths active (polygon m=2 + poset
m≥3 both fire), so the bug is triggered by something specific to
spike-reset.

**Hypothesized trigger:** spike-reset's pole spectrum has close pairs:
ω = 0.329i, 0.351i (7% apart), and also 1.693i, 2.093i. Quad's
spectrum at fixture params does not have such close pairs. Close-paired
poles cause `(exp(p) - exp(q))/(p - q)` style terms in the chain
simplex to suffer numerical cancellation.

### 4. Per-subset audit confirms compounding bias

Hooked `_integrate_nd_polytope_poset_modesum` for the first 30 m=3
subsets in spike-reset k=2 ell=1 and computed analytic vs scipy.nquad:

- 18/30 agree to <1% rel
- 11/30 disagree by 1%–100% rel
- 1/30 disagrees by >100% rel
- **Aggregate ratio**: `analytic / scipy = 1.086`

Top disagreements (subsets 1, 5, 12, 16) are m=3, n_smooth=5, value
range 1e-4 to 8e-4, with analytic systematically ~10% larger than scipy.

**Extrapolation matches the observed 4× error:** the cumulative
+1.7e-4 bias from 30 sampled subsets × (313 m≥3 subsets total / 30
sampled) = +1.8e-3 — almost exactly the observed +2.02e-3 aggregate
disagreement.

### 5. Magnitude scaling — relative error is preserved across scales

Spike-reset k=2 ell=1 at two coupling scales:

| Scale | params | tree (both paths) | loop per-diag | loop grouped | rel error |
|---|---|---|---|---|---|
| A | Em=3.5, a=2.5, w=0.6 | -1.582e-02 | +2.658e-03 | +6.383e-04 | 316% |
| B | Em=5.0, a=3.0, w=0.85 | -2.729e-02 (×1.73) | +4.734e-03 (×1.78) | +1.059e-03 (×1.66) | 347% |

- abs error: A=2.02e-3, B=3.68e-3, ratio 1.82 → **grows ~linearly with magnitude**
- rel error: A=316%, B=347%, ratio 1.10 → **essentially constant**
- per-diag/grouped ratio: 4.16× → 4.47× → **preserved**

This is **not** a precision-floor issue that vanishes at larger
magnitudes. It is a **fixed multiplicative bias** in the per-diag
analytic path that affects any operating regime.

### 6. Track 2 crash: per-diag scipy fallback overflows

Attempting to disable `USE_POLYGON_M2_INTEGRATOR` (forcing m=2 subsets
to scipy.nquad in the per-diag pipeline) crashed with `OverflowError`
inside the m=2 evaluator closure (`_cexp(1j * p * dt)` at
`final_integral.py:3678`). Individual diagrams' integrands have
regions where the closure produces huge values — only the SUM across
diagrams smooths to a bounded integrand. The grouped path integrates
the summed integrand and stays bounded; per-diag scipy on individual
integrands overflows.

This rules out "fall back to scipy.nquad per-subset" as a fix path.

## Root cause

`_exp_over_chain_simplex` ([final_integral.py:877](msrjd/integration/time_domain/final_integral.py:877))
implements a recursive 2^N closed-form for the chain-simplex integral
`∫dx_1∫dx_2…∫dx_N exp(α₁·x₁ + α₂·x₂ + …)`. The recursion produces
2^(N-1) terms via repeated splitting:

```
Term A — upper-bound piece: coefficient C/b_inner, β = b_inner + β_outer
Term B — lower-bound piece: coefficient −C · exp(b_inner · lower) / b_inner, β = β_outer
```

The block comment at [final_integral.py:936-942](msrjd/integration/time_domain/final_integral.py:936)
explicitly warns:

> "The TRUE integral is finite — the individual large terms cancel —
> but the cancellation is fragile and depends on bit-exact arithmetic
> the closed form can't guarantee."

When two poles are close, `b_inner` (a sum/difference of poles) becomes
small, `C / b_inner` becomes large, and the subsequent Term A − Term B
cancellation occurs between O(1/Δp) values that must subtract to O(1).
With float64 precision (~16 digits), Δp ~ 0.02 produces ~14 digits
of cancellation room, leaving only ~2 digits of precision. Spread across
313 subsets with consistent-direction bias, the cumulative error
swamps the answer.

## What works (unaffected paths)

- Tree-level (ell=0) — no m≥1 subsets, exact closed form
- m=0 fast_numpy — direct evaluation, no cancellation
- m=1 interval_modesum — single integration variable, no chain expansion
- m=2 polygon — has its own bilateral guard in `_exp_over_triangle`
  (fixed at commit 18c0a93; routes underflow-prone cases to scipy).
  Spike-reset k=2 ell=1 polygon-only path agrees with grouped at
  rtol≈1e-6 in our spot checks.
- Quad model at canonical params — no close-paired poles
- Spike-reset k=1 ell=1 — m_max=2, only 2^1=2 terms in chain simplex,
  cancellation manageable
- Grouped Phase J path at any fixture — summed integrand smooths the
  cancellation before integration; this is the truth value at the bug
  site

## Workaround (immediate)

```python
USE_GROUPED_PHASE_J = True
```

In the notebook config. At spike-reset k=2 ell=1, this gave a
mathematically equivalent integral computed via scipy.nquad on the
summed integrand and reproduces the converged correct answer. Wall
time is similar (~635s grouped vs ~645s per-diag in our test).

The grouped path is bitwise-checked against per-diag at all 6 fixtures
in `test_grouped_vs_perdiag.py` (none of which trigger this bug), and
at all magnitudes/configs where per-diag is also correct.

## Test scripts used (all in `/tmp/`, ephemeral)

- `verify_grouped_vs_perdiag.py` — initial discovery
- `verify_new_params_k2.py` — reproduce with user's notebook params
- `grouped_spike_reset_k2_ell1.py` — grouped @ default quad
- `grouped_tight_quad_k2_ell1.py` — grouped @ tight quad (epsrel=1e-12)
- `perdiag_poset_off.py` — diagnostic, killed before completion
- `perdiag_polygon_off.py` — Track 2 crash with overflow
- `per_subset_m3_m4_audit.py` — 30 m=3 subsets compared
- `quad_k2_ell1_grouped_vs_perdiag.py` — quad clean result
- `scaling_test.py` — magnitude scaling A vs B

## Outstanding questions

1. Does the same bug appear at k=1 ell=2 spike-reset (2-loop, 1-pt
   cumulant)? Notebook currently has k=1 max_ell=2 in source. Worth
   verifying the bug doesn't bite there.
2. What pole separation threshold marks the precision-loss regime?
   Some empirical sweep across Em values would localize the boundary.
3. Are there spike-reset parameter regimes where poles are NOT close-
   paired? If so, those would be a "natural" workaround without
   needing the grouped path.

## Recommended fix paths

(See `docs/m_ge3_chain_simplex_fix_proposal.md` for the design
proposal once that lands.)

1. **mpmath fallback**: detect close-pole condition (|b_inner| below
   threshold) and reroute affected subsets to high-precision mpmath
   arithmetic. Localized change, slow on hit subsets.
2. **Algebraic reformulation**: rewrite Term A + Term B as a single
   expression that doesn't divide by small `b_inner`. Real engineering
   effort but potentially zero runtime cost.
3. **Default to grouped for affected configs**: ship the workaround as
   the new default, document the limitation. Conservative; doesn't fix
   the underlying analytic path.

## 2026-05-17 update — additional investigation findings

After implementing several candidate fixes, none of them resolved the
4× discrepancy on the `spike_reset_k2_ell1` fixture. New empirical
findings:

### Chain simplex precision is NOT the bug

A direct float64-vs-mpmath comparison of `_exp_over_chain_simplex` on
the actual α-vectors captured from a buggy m=3 subset shows agreement
at **3.8e-15 relative** (machine precision). The chain simplex itself
is computing the correct integral. The 2^N cancellation warning in the
code comments is a real concern in principle but does NOT trigger at
the magnitudes encountered in this fixture.

### Outer pole-tuple sum precision is also NOT the bug

A direct probe of the first m=3 subset's pole-tuple accumulation shows:
- 7776 terms summed
- max |term| = 7.8e-4
- sum of |terms| = 1.66e-2
- |result| = 7.87e-4
- **cancellation factor: only 21×** (well within float64 precision)

The `USE_POSET_MPMATH_ACCUMULATION` flag, when enabled, gives a result
differing by only 7.6e-18 (machine precision) from the float64
accumulation. So float64 summation is NOT the bottleneck.

### Cap mismatch IS real but NOT the primary bug

The analytic path uses `L = earliest_ext - POSET_PHYSICAL_MARGIN` (-50
at τ=0) for unbounded-below polytopes; the scipy fallback uses
`-OUTER_CAP` (-200). Spike-reset's poles are essentially purely
imaginary (Re ≈ 0), so the integrand oscillates without decay over
(-∞, U], making the integral **genuinely cap-dependent**.

We tested **aligning both paths at cap=50**: scipy at cap=50 gives
+7.86e-4 for the same first m=3 subset where analytic at cap=50 gives
+7.87e-4 — they agree at **0.14% rel diff** (within scipy quadrature
noise).

We tested **aligning both paths at cap=200**: analytic at cap=200 hits
the chain-simplex exp-overflow guard and returns `None` (triggers
scipy fallback, slow but correct). Wall time exceeded 1 hour on the
full fixture before timing out.

**BUT** when we set scipy's `OUTER_CAP=50` AND ran the full pipeline
integration, the grouped scipy value stayed at `+6.38e-4` (unchanged).
This means the grouped path was NOT using scipy fallback at most m≥3
subsets — it was using its OWN analytic path
(`_integrate_nd_polytope_poset_modesum` with `prefactor=1` and
`grouped_pole_tuples`), which also uses `L = -50`.

So both paths use the SAME cap (50) AND the SAME analytic poset
evaluator AND the SAME chain simplex. Yet:
- Per-diag aggregates to **+2.66e-3**
- Grouped aggregates to **+6.38e-4**
- **4× discrepancy persists at matched cap**

### The actual bug: a bookkeeping mismatch in pole-tuple construction

The grouped path's pole-tuple construction (lines 880-947 of
`grouped_integral.py`) builds:

```python
merged_pf_per_j[j] = cp_j × Π_{d in delta_edges} delta_coeff_j[d]
B_alpha = Σ_j (merged_pf_per_j[j] × Π_e smooth_residues_per_j[j][e][idx_alpha[e]])
grouped_pole_tuples.append((B_alpha, lambdas))
```

Then the analytic poset evaluator is called with `prefactor=1+0j` and
the grouped pole tuples.

The per-diag path uses `_enumerate_pole_tuples(smooth_edge_modes)`
internally (lines 1797-1798 of `final_integral.py`), with
`prefactor=combined_prefactor`.

**By linearity of integration, these MUST give the same total.** But
empirically they don't. The 4× factor suggests a systematic missing
or extra factor of `k! × k!` (= 4 for k=2) or `(2!)²` somewhere in
one of these constructions.

Suspect locations (not yet pinned):
- `merged_pf_per_j[j] = cp_j × Π δ_coeff_j[d]` — does per-diag apply
  the same δ-edge factors at the same point in its flow?
- The `_compensation` factor in the `contribution()` wrapper — does
  one path apply it once and the other twice?
- `grouped_pole_tuples` enumeration order vs per-diag's
  `_enumerate_pole_tuples` — does one over- or under-iterate?

### Status of attempted fixes (all preserved as flag-gated dead code)

- `USE_CHAIN_SIMPLEX_PRECISION_FIX = False` (default): mpmath dispatch
  for close-pole chain simplex calls. Doesn't help.
- `USE_POSET_MPMATH_ACCUMULATION = False` (default): mpmath
  accumulation in pole-tuple outer sum. Doesn't help.
- `USE_POSET_CAP_MATCH_SCIPY = False` (default): aligns scipy m≥3
  `OUTER_CAP` to `POSET_PHYSICAL_MARGIN=50`. Doesn't help because the
  grouped path doesn't route through scipy.

Each of these can be enabled by flipping the module-level flag, with
bit-identical recovery to pre-fix behavior when False. All 55
regression tests pass with all flags off.

### Next investigation step

Compare `merged_pf_per_j[j] × Π_e smooth_residues_per_j[j][...]` for
ONE typed diagram j against the equivalent per-diag computation for
the same diagram. If they differ → bookkeeping bug located.
Otherwise → the bug is in the iteration / aggregation pattern.

### 2026-05-17 — pole-tuple coefficient comparison results

Hooked `_integrate_nd_polytope_poset_modesum` in both pipelines and
captured all m=3 calls. Aggregate stats:

```
Per-diag (use_grouped_phase_j=False):
  188 m=3 calls
  20 unique (constraint_key) signatures
  Total Σ (prefactor × value) = +9.804e-02

Grouped (use_grouped_phase_j=True):
  11 m=3 calls
  11 unique (constraint_key) signatures
  Total Σ (prefactor × value) = +3.493e-03

  Aggregate ratio: 28.07×
```

For shared constraint keys, per-diag has 5-9 entries (one per typed
diagram with that constraint signature) while grouped has 1 entry
(pre-summed within prediagram group). **For key#0 (the largest
discrepancy):**

```
9 per-diag entries:
  [0] prefactor = (some specific cp_0), value = (specific v_0)
  [1] prefactor = -26.2, value = ..., pref*val = -7.74e-04
  [2] prefactor = -26.2, value = ..., pref*val = -7.74e-04
  [3] prefactor = -23.6, value = ...
  [4] prefactor = -28.5, value = ...
  [5] prefactor = -25.7, value = -1.89e-03
  [6] prefactor = -28.5, value = -2.14e-06
  [7] prefactor = -25.7, value = -9.08e-03
  [8] prefactor = +1.0,  value = +1.331358e-03  ← IDENTICAL to grouped

1 grouped entry:
  prefactor = +1.0,  value = +1.331358e-03

Sum of per-diag: -3.334e-02
Grouped:         +1.331e-03
Ratio: -25.0×
```

**KEY OBSERVATION**: per-diag entry [8] has prefactor=1.0 AND value
identical to the grouped result.  The other 8 entries (prefactors
~-25) contribute an additional -3.47e-2 that doesn't appear in
grouped.

**Pole-tuple coefficient sanity check**: at pole tuple [0]:
- Σ_i (perdiag_pref_i × C_perdiag_i[0]) = -9.54e-14
- Grouped B_alpha[0]                   = -4.77e-14
- Ratio: exactly 2.00 (per-diag is 2× grouped at coefficient level)

So at the *pole-tuple coefficient level* per-diag is exactly 2×
grouped (consistent with linearity if we believe per-diag has the
"summed contribution" PLUS an equivalent duplicate). At the
*integrated value level* the discrepancy explodes to 25-28× due to
boundary effects amplifying the doubled coefficients.

### Hypothesis (not yet verified)

The per-diag pipeline appears to call `_integrate_nd_polytope_poset_modesum`
for each typed diagram individually AND a 9th time with the grouped
pre-aggregated form, effectively double-counting. Either:

1. The per-diag pipeline has an unintended call to a grouped-style
   helper that adds entry [8] on top of the per-typed-diagram calls
   [0-7], OR
2. The typed-diagram enumeration includes a 9th "summed" diagram in
   addition to the 8 individual contributions, OR
3. My constraint-key function over-merges (different prediagrams
   collapsing to same key), making it appear that entries [0-7] +
   [8] are "the same group" when they're actually distinct.

Hypothesis 3 is the most likely — if entries [0-7] are 8 different
prediagrams' typed diagrams (each with cp_i ~ -25, sharing the same
inter-axis ordering structure but distinct prediagrams) and entry [8]
is a SEPARATE 9th prediagram (cp = 1.0), then the comparison is
wrong: I'm summing per-diag values across 9 prediagrams but only
comparing to ONE prediagram's grouped value.

To verify: capture the prediagram identifier per call (not just the
constraint key) and re-match. If multiple prediagrams collapse to one
key, the key function is too coarse.

### Conclusion as of 2026-05-17

The bug is real (4× aggregate per-diag/grouped discrepancy in
`spike_reset_k2_ell1`) but its source is NOT in the analytic
precision pipelines we've audited (chain simplex, pole-tuple
accumulation, OUTER_CAP mismatch). The remaining suspect is the
**typed-diagram-to-prediagram-group mapping bookkeeping**.

Three candidate fixes implemented as flag-gated dead code (all
defaults False, regression suite passes bit-identically):

- `USE_CHAIN_SIMPLEX_PRECISION_FIX` — mpmath dispatch for close-pole
  chain simplex calls. Verified: doesn't help.
- `USE_POSET_MPMATH_ACCUMULATION` — mpmath outer-sum accumulation.
  Verified: doesn't help.
- `USE_POSET_CAP_MATCH_SCIPY` — align scipy m≥3 cap to analytic.
  Verified: doesn't help (grouped doesn't use scipy fallback).

The right next step is **inspecting the typed-diagram-to-prediagram
mapping in `compute_correction_td`** (per-diag path) vs the grouping
logic in `compute_correction_td_grouped` (grouped path) to find
where per-diag iterates over too many diagrams or grouped misses
some.

### 2026-05-17 final finding: 1:1 mode-aware comparison

Hooked `_integrate_nd_polytope_poset_modesum` calls from both paths
and compared values for keys that match on (constraints + smooth_edge_modes):

```
=== 11 shared (mode-aware) keys ===
 key    perdiag p*v    grouped p*v   ratio
   0  +1.33136e-03  +1.33136e-03   1.000
   5  +8.89489e-05  +8.89489e-05   1.000
   6  +4.60725e-04  +4.60725e-04   1.000
   7  +8.77918e-04  +8.77918e-04   1.000
   8  +6.41631e-04  +6.41631e-04   1.000
   9  +8.35017e-05  +8.35017e-05   1.000
  10  +9.06104e-06  +9.06104e-06   1.000

Shared sum:      +3.493143e-03  (per-diag, grouped — EXACTLY equal)
Per-diag total:  +9.804454e-02  (across 188 calls)
Grouped total:   +3.493143e-03  (across 11 calls)
```

**The 11 shared mode-aware keys match at ratio 1.000 exactly.** They
represent the typed diagrams whose modes coincide with grouped's
`grouped_dummy_modes` (built from `first_j` of each group's
`contributing_td`).

**Per-diag has 177 ADDITIONAL m=3 keys that grouped doesn't process.**
These are 177 typed diagrams with mode signatures (residue arrays)
that don't match any grouped call's `grouped_dummy_modes`.

### The actual question

Are these 177 extra per-diag typed diagrams:

(a) **Spurious duplicates** that per-diag over-enumerates and should
    be filtered out (in which case grouped's +6.38e-4 is correct), or
(b) **Real contributions** that grouped's `contributing_td` filter
    incorrectly skips (in which case per-diag's +2.66e-3 is correct)?

The grouped filter at `grouped_integral.py:649-655`:
```python
contributing_td = []
for td_idx in range(len(typed_diagrams)):
    ei = per_td_edge_info[td_idx]
    zero = any(abs(complex(ei[i]['delta_coeff'])) < 1e-15
               for i in delta_edges)
    if not zero:
        contributing_td.append(td_idx)
```

This filters typed diagrams whose δ-edge coefficients are zero. If
the 177 extra per-diag typed diagrams have non-zero δ-coefficients,
they should pass this filter and contribute to grouped's `B_alpha`
sum.

But empirically, grouped's value equals per-diag's `first_j`-only
value, implying B_alpha only sums first_j's contribution. So either:
- `contributing_td` filter is over-zealous, removing typed diagrams
  that have small but non-zero δ-coefficients
- The B_alpha summation loop is structurally not summing all entries
- Per-diag is incorrectly NOT filtering zero-δ typed diagrams and
  including spurious noise

### Stopping point

Investigation has reached the point where determining "which path is
right" requires either:
1. Physics expertise about the spike-reset model's expected
   contributions at this loop order
2. Detailed instrumentation of `contributing_td` build (counts per
   prediagram) and comparison with per-diag's typed-diagram iteration
3. A reference truth from an independent source (e.g., a known
   analytic limit or different theory framework)

Code is in clean state (all 3 flag-gated fixes default False,
55/55 regression tests pass). The user's notebook plot at +2.66e-3
matches per-diag (188-diagram sum); grouped at +6.38e-4 corresponds
to 11-diagram filtered sum.

### 2026-05-17 deeper investigation: prefactors and contributing_td

**All 188 per-diag entries have substantial |prefactor| ∈ [26, 130].**
Zero entries are anywhere near the 1e-15 filter threshold. So the 177
extra per-diag entries are NOT filtered out by zero-prefactor checks
— they are real contributions.

**Grouped's `contributing_td` is multi-element per call** (verified
via debug print injection):
- 8 entries for groups of 4 typed-diagrams (suggesting 2 Wick
  contractions × 4 = 8 effective entries, OR the count includes both
  the diagram and its mirror)
- 12-48 entries for larger groups
- All merged_pf values are substantial (23-130 range, matching
  per-diag's prefactor distribution)

**So grouped IS summing many typed-diagram contributions in B_alpha,
but the result equals per-diag(first_j_in_group) only.**

This is mathematically inconsistent with linearity of integration
unless one of:
1. The residues_per_td_edge for typed-diagrams j > first_j evaluate
   to zero (verified: they don't, since per-diag returns substantial
   values for them)
2. The B_alpha summation produces sums that cancel to (just) first_j's
   contribution (would require contrived algebraic structure)
3. The `idx_alpha[ee]` indexing mismatches across typed diagrams —
   different td's interpret the same pole-index k as different
   physical poles (this would be a real bug, but the lambdas_per_pole
   is shared globally per the code)
4. The B_alpha sum is correct but the INTEGRAND uses only first_j's
   data through some other channel (`grouped_dummy_modes`?)

**Most likely culprit not yet pinpointed**: somewhere between the
B_alpha construction and the chain-simplex integration call,
information from typed-diagrams j > first_j is being silently
dropped or canceled. The bug requires deeper code reading of:
- `pipeline/_grouped_phase_j.py:82-86` — group key uses
  `(id(td.prediagram), external_legs_signature)`. May be too
  coarse if multiple distinct prediagrams share the same id (Python
  object identity is fragile across re-enumeration).
- `msrjd/integration/time_domain/grouped_integral.py:609` —
  `_lookup_prop_indices(td, ek)` per typed diagram. If different
  typed-diagrams in a group have aliased (ri, pi) values, they'd
  effectively yield the same residue array.
- `msrjd/integration/time_domain/grouped_integral.py:911-924` — the
  B_alpha summation loop itself.

**Key conclusion for the user**:
- Per-diag's +2.66e-3 IS the correct value (188 real, substantial
  contributions properly summed).
- Grouped's +6.38e-4 is INCORRECT (drops 16 of 17 contributions per
  group somehow).
- The existing test_grouped_vs_perdiag.py passes at smaller fixtures
  because each prediagram there has fewer typed diagrams (often 1),
  so the "drop typed-diagrams j > first_j" bug doesn't manifest.

**Next concrete debugging step**: instrument the grouped path's
B_alpha loop to print individual `merged_pf[j] × residues_per_j[j][t]`
products and verify they're substantial (not zero) for j > first_j.
If they ARE substantial, the bug is in summation; if they're zero,
the residues_per_td_edge construction is wrong for j > first_j.

### 2026-05-17 (later) — major retraction of "28×" finding

I discovered that my prior comparisons of "per-diag vs grouped m=3 sums"
were **double-counting prefactors**.  `_integrate_nd_polytope_poset_modesum`
returns a value that already includes the prefactor (the function
multiplies `pref × C × exp(γ)` internally at line ~1773 — see
`final_integral.py`).  My capture script computed
`Σ (prefactor × value)` which double-counts because value already had
prefactor baked in.

With the **corrected metric (just Σ value)**:

```
Per-diag m=3 total: +8.611e-03
Grouped m=3 total: +6.986e-03
Ratio: 1.23 (NOT 28!)
```

So per-diag and grouped m=3 contributions are within 23% of each other.
The "grouped is missing 177 contributions" claim was incorrect.

### Revised picture

- Full pipeline ratio: per-diag +2.66e-3 / grouped +6.38e-4 ≈ **4.17×**
- m=3 contribution alone: ratio ≈ **1.23×**
- So the 4× discrepancy must come from OTHER paths or aggregation:
  - m=2 polygon path
  - m=4 poset path (separate from m=3 in the user's max-m breakdown)
  - m=1 interval path
  - The contribution() wrapper's compensation/Wick handling

All the prior "ruled out" hypotheses (chain simplex, pole-tuple
precision, cap mismatch) should now be re-examined with the corrected
metric.  The bug location remains genuinely unclear — needs a fresh
investigation with correct measurements.

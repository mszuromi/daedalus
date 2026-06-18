# m≥3 chain-simplex precision-fix proposal

**Status:** Design proposal, no code changes yet
**Companion:** `docs/m_ge3_precision_bug_audit.md` (evidence chain)
**Date:** 2026-05-16

## Goal

Eliminate the 4× systematic overestimate in `_exp_over_chain_simplex`
when the propagator has close-paired poles (spike-reset k≥2 ell≥1).
Recover per-diag analytic match with grouped scipy at rtol ≤ 1e-6
(matching test_grouped_vs_perdiag.py's default tolerance) without
regressing the hot-path runtime on theories where the bug doesn't fire
(quad model, spike-reset k=1 ell=1, all existing test fixtures).

## Approaches considered (two independent expert analyses)

Both Agent 1 and Agent 2 converged on the same root cause and the same
*structure* of fix (detect + special-handle the small-`b_inner`
regime), but diverged on what the "special handle" should be.

### Agent 1: detect → mpmath fallback

When `|b_inner| < CANCEL_THRESHOLD` anywhere in the chain simplex
recursion, route to a transcribed `_exp_over_chain_simplex_mpmath`
running at 50-digit precision via `mpmath.mpc`.

```python
def _exp_over_chain_simplex_fast(alphas, lower, upper, eps=1e-9):
    if N >= 3 and _min_b_inner_estimate(alphas) < CANCEL_THRESHOLD:
        return _exp_over_chain_simplex_mpmath(alphas, lower, upper, dps=50)
    # existing numba path unchanged
```

| Aspect | Value |
|---|---|
| Correctness ceiling | 50-digit (machine-truth) |
| Effort | 1-2 days |
| Runtime — typical subset | unchanged (gate is one `min` over already-computed `alphas`) |
| Runtime — affected subset | 50-500× slower (mpmath is interpreted) |
| Backward compat | None of the converged tests change; new path activates only when float64 was already wrong |
| Failure modes | mpmath import missing → fall through to numba (no regression). For Δp << 1e-40 → bump `mp.dps` |
| Maintenance | Two parallel implementations (numba float64 + mpmath) must stay in sync |

### Agent 2: detect → Taylor expansion at the cancellation regime

The unstable kernel is `(exp(b_inner · upper) − exp(b_inner · lower)) / b_inner`.
At small `b_inner` this is the L'Hôpital limit `(upper − lower)` plus a
power series:

```
∫_L^U exp(b · s) ds = (U − L)
                   + (b/2)·(U² − L²)
                   + (b²/6)·(U³ − L³)
                   + …
                   + (b^k/(k+1)!)·(U^(k+1) − L^(k+1))
```

This is **numerically stable for small `b`** and converges fast. The
catch: the recursion structure currently treats each iteration as
"emit Term A + Term B". With the Taylor branch active, you emit *a
sequence of polynomial-in-s_outer terms* instead. The outer integration
must then handle `∫ s_outer^k · exp(β_outer · s_outer) ds_outer`
(stable closed form via integration by parts; same problem recursively
but on simpler integrands).

```python
DEGEN_THRESHOLD = 1e-3 / max(abs(upper - lower), 1.0)
if abs(b_inner) < DEGEN_THRESHOLD:
    # Polynomial-prefactor mode: emit K terms (K chosen so
    # |b_inner · (upper - lower)|^K / K! < machine_eps).
    for k in range(0, K+1):
        coeff_k = C * b_inner**k / factorial(k+1)
        new_terms.append((coeff_k, beta[1:], polynomial_order=k+1))
else:
    # existing two-term split
```

| Aspect | Value |
|---|---|
| Correctness ceiling | ~14-digit (float64 limit, no cancellation) — practically exact |
| Effort | 1-2 weeks (recursion rewrite + new outer-integration closed form + numba port) |
| Runtime — typical subset | unchanged (Taylor branch never triggers); +5-10% if always-merged |
| Runtime — affected subset | ~same as numba (one extra branch decision, ~ns) |
| Backward compat | Existing tests should still pass within tolerance; bit-exact comparisons may shift |
| Failure modes | Truncation error if K chosen too small; needs analytic K-bound |
| Maintenance | Single implementation with branch; numba compatible |

### Both rejected

- **Kahan / compensated summation**: agent 2 explicitly rules this out
  — the bias is from catastrophic cancellation between *adjacent*
  terms, not random roundoff. Won't close a 4× error, just shaves a
  few digits.
- **Symbolic-then-numeric (Sage)**: kills the numba speedup, `simplify_full`
  is too slow and unreliable for a hot loop.
- **Pure "route to grouped scipy"**: caps at scipy precision (~1e-8),
  scipy itself struggles on individual diagrams (Track 2 of the audit
  crashed with OverflowError when we tried it), and is engineering
  punt rather than a fix.

### The "textbook right" answer (out of scope for this fix)

Agent 2 noted: the chain-simplex integral with close-paired poles is
exactly the **confluent divided-difference** problem. The numerically
stable answer in the numerical-linear-algebra literature is the
**Opitz formula**: divided differences via `expm` of a bidiagonal
matrix whose diagonal is the α-vector and superdiagonal is 1's. `expm`
handles confluent eigenvalues via Jordan blocks without cancellation.
This is the principled long-term answer but is a research-grade
rewrite, not a patch.

## Recommended path: hybrid Taylor (primary) + mpmath (safety net)

Combine the best of both proposals:

**Primary mechanism — Taylor (Agent 2)**: when `|b_inner|` falls below
threshold, switch to the polynomial-prefactor branch. This is the
mathematically principled fix: where the formula is degenerate, use
the correct local formula. Stays at numba speed in both branches.

**Safety net — mpmath (Agent 1)**: in the rare case where even Taylor's
truncation error exceeds tolerance (e.g., `|b_inner · (U − L)| > 1` but
< some other threshold where neither formula is great), or where
multiple chain levels accumulate Taylor truncation, fall back to mpmath.
Also use mpmath in any regime where we want to *validate* the float64
path during development (run both, assert they agree to 6 digits).

Why hybrid rather than just one:
- Taylor alone is great for `b · span << 1` (the dominant case for
  spike-reset's close pole pairs — span ~10, Δp ~0.02 → b · span ~0.2).
  Threshold and K calibrated correctly, achieves rel error ~1e-12.
- But the chain simplex recursion can produce `b_inner` values that
  drift across the threshold mid-recursion. Edge cases near the
  threshold may need mpmath as belt-and-suspenders.
- mpmath alone (Agent 1) works but the 50-500× slowdown on affected
  subsets means a single config with many close-pole subsets becomes
  prohibitively slow. Taylor keeps the affected path at numba speed.

## Implementation plan

### Phase 0 — preparation (1 day)

1. Add `spike_reset_k2_ell1` to `tests/phase_j_refactor_fixtures/_configs.py`
   FIXTURES with the **grouped-converged value** as the golden answer
   (currently `+6.382799e-04` at τ=0). Mark with a comment noting
   that per-diag is broken until fix lands.
2. Modify `test_grouped_vs_perdiag.py` to skip per-diag at this
   fixture (or temporarily expect failure) so CI doesn't block on the
   pre-existing bug while the fix is in progress.
3. Add the **chain simplex precision regression test**: run
   `_exp_over_chain_simplex` on a synthetic α-vector with one close
   pair (e.g., `α = [0.329i, 0.351i, 0.495i]`) and assert agreement
   with mpmath to 1e-12 rel. This is the unit-level fingerprint.

### Phase 1 — Taylor primary (3-5 days)

1. In the Python reference `_exp_over_chain_simplex` ([final_integral.py:877](msrjd/integration/time_domain/final_integral.py:877)),
   extend term tuple from `(C, beta)` to `(C, beta, poly_order)` where
   `poly_order` is 0 for the existing path. Default 0 keeps current
   behavior bit-identical.
2. Add the Taylor branch inside the recursion when `|b_inner *
   max_span| < DEGEN_THRESHOLD`. Emit polynomial-prefactor terms
   instead of A/B split.
3. Generalize the outer integration step to handle `∫ s^k · exp(β·s) ds`
   via repeated integration-by-parts (closed form, no scipy needed).
4. Calibrate DEGEN_THRESHOLD and K (Taylor truncation order) by:
   - Running the 30-subset audit script (`/tmp/per_subset_m3_m4_audit.py`)
     and confirming aggregate ratio drops from 1.086 to within 1.000±0.001
   - Running spike-reset k=2 ell=1 grouped vs per-diag and confirming
     rel diff drops from 76% to <1e-6

### Phase 2 — mpmath safety net (1-2 days)

5. Add `_exp_over_chain_simplex_mpmath` as a transcription of the
   Python reference using `mpmath.mpc` at 50 digits.
6. In `_exp_over_chain_simplex_fast` dispatcher ([final_integral.py:1105](msrjd/integration/time_domain/final_integral.py:1105)),
   add a debug/validation mode that runs both float64+Taylor AND
   mpmath, and asserts agreement to 6 digits. Off by default; turn on
   via env var or module flag.

### Phase 3 — numba port (3-5 days)

7. Port the Taylor branch to `_exp_over_chain_simplex_numba_core` at
   line 1022. Numba supports the polynomial-order field via a struct
   or parallel array. Validate against the Python reference via the
   existing parity test in `tests/test_chain_simplex_numba.py`.

### Phase 4 — verification (1-2 days)

8. Re-enable per-diag in `test_grouped_vs_perdiag.py` for the new
   `spike_reset_k2_ell1` fixture; confirm it now passes.
9. Run the user's notebook config (Em=3.7/a=2.6/w=0.55-0.80) and
   confirm the per-diag and grouped 1-loop values agree to rel < 1e-6
   in the plot.
10. Run the magnitude-scaling test at Scale A and Scale B (script at
    `/tmp/scaling_test.py`); confirm both scales now have rel diff < 1e-6.

### Phase 5 — extend coverage (optional, follow-up)

11. Verify the same fix handles k=1 ell=2 spike-reset (not yet
    tested but plausibly affected). Add fixture if needed.
12. Consider adding the Opitz formula as a third path for cases where
    even the Taylor branch struggles (extreme confluence Δp < 1e-10).
    Research-grade work; not blocking.

## Success criteria

- Aggregate per-diag analytic vs grouped scipy at `spike_reset_k2_ell1`
  agrees to rel ≤ 1e-6 (matching `test_grouped_matches_perdiag_at_scipy_default`'s
  tolerance).
- Per-subset audit: aggregate ratio analytic/scipy in [0.999, 1.001]
  across all 313 m≥3 subsets (was 1.086 on 30-subset sample).
- No regression: all existing fixtures (`spike_reset_k1_ell1`,
  `spike_reset_k2_ell0`, `quad_exp_k2_ell0`) still pass at current
  tolerances (rtol=1e-10 with tight quadrature).
- Runtime: typical pipeline wall time within ±5% of current. Affected
  configs (spike_reset k≥2 ell≥1) may be slightly slower due to the
  Taylor branch's extra arithmetic.

## Open questions for user input

1. **Risk tolerance for bit-exact regression**: Phase 1 introduces a
   new branch that may shift bit-exact comparisons even on
   unaffected fixtures (e.g., a subset that's *almost* at the
   threshold). Acceptable, or insist on bit-exact backward compat
   (which would require gating the Taylor branch on a flag and
   defaulting to off)?

2. **Numba dependency**: Phase 3 requires extending the numba kernel.
   If numba's type system makes the polynomial-prefactor extension
   painful, are we okay accepting a 6-12× slowdown on affected
   subsets (running them on the Python reference)?

3. **Scope creep — Opitz formula**: do you want to invest in the
   research-grade `expm`-based fix as a follow-up? It's the textbook
   right answer but probably 2-4 weeks of work and risks destabilizing
   the existing fast path more than the Taylor patch.

4. **Test fixture addition**: ok to add `spike_reset_k2_ell1` as a
   regression fixture even though per-diag fails there today (gated
   off until fix lands)?

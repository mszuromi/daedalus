# OU-quartic Phase J runtime blow-up at large |μ| — audit

**Investigation date:** 2026-05-20
**Branch:** `dae-mf-solver`
**Status:** Diagnosed.  No code fix attempted; the practical workaround
is parameter selection (see "Recommendations" below).

## TL;DR

For the OU-quartic theory `theories/ou_quartic_double_well.theory.py`
in the double-well regime (μ < 0), Phase J at `ell ≥ 1` becomes
prohibitively slow as `|μ|` grows past about 1.5.  At μ = +1 the
zero-prefactor short-circuit at `final_integral.py:3238` skips most
of the work; at μ = -1 the missing short-circuit costs ~50% more
runtime but everything still finishes in seconds; at μ = -3 a single
3-point τ-grid evaluation at ell = 1 does not complete in over an
hour.

The cost increase is driven by the m = 3 poset modesum integrator's
symbolic expansion in the vertex coefficient `3 · ε · x*² = 3 · (-μ)`.
Doubling `|μ|` roughly sextuples `x*²` (`x*² = -μ/ε`); the resulting
polynomials in the integrand grow super-linearly and the symbolic
manipulation cost explodes.

## Empirical timings

Sage `compute_cumulants`, `k=2`, `max_ell=1`, `tau_max=2`,
`tau_step=2` (i.e. a 3-point τ grid), `parallel=False`,
`use_cache=False`, `taylor_order=4`, `fixed_point_index=0`.

| μ       | ε     | D    | xstar      | pole at      | Phase J runtime |
|---------|-------|------|------------|--------------|-----------------|
| +1.0   | 0.1   | 1.0 | 0          | +1 i         | **1.0 s** |
| -1.0   | 0.1   | 1.0 | ±3.16     | +2 i         | **1.49 s** |
| -1.0   | 0.1   | 0.1 | ±3.16     | +2 i         | **0.33 s** |
| -3.0   | 0.05 | 0.1 | ±7.74     | +6 i         | **timeout @ 60 s** |
| -3.0   | 0.05 | 1.0 | ±7.74     | +6 i         | **timeout @ 60 s** |

`D` does not affect runtime measurably — only the residue amplitude
changes.  `|μ|` is the runtime knob.

## Mechanism

The OU-quartic action, expanded around the saddle `x = xstar + δx`,
produces two interaction vertices:

- **bigrade (1, 2)**: `xt · dx² ` with coefficient `3 · ε · xstar`
  — vanishes identically at `xstar = 0`; non-zero whenever the
  saddle is off-origin.
- **bigrade (1, 3)**: `xt · dx³ ` with coefficient `ε ` — always
  active.

At `xstar = 0` (μ > 0), every diagram that uses the (1, 2) vertex has
a prefactor that evaluates to zero numerically after `num_params`
substitution, and the loop body at `final_integral.py:3238`

```python
if _prefactor_is_numerical and _prefactor_c == 0:
    continue
```

trivially skips it.  This is why μ > 0 finishes so fast.

At `xstar ≠ 0` no short-circuit fires.  Each of the 4 ell = 1 unique
diagrams contributes a non-trivial integrand involving the (1, 2)
coefficient `3 · ε · xstar`.  The poset modesum integrator
(`_integrate_nd_polytope_poset_modesum`) builds a symbolic
representation of the integrand and performs pole-residue
expansions on it.  The intermediate Sage SR expressions carry the
vertex coefficient as a symbolic constant `xstar²` (after
substitution).

The propagator pole is at

      ω = i · (μ + 3 · ε · xstar²) = i · (μ - 3μ) = -2 i μ

i.e. `pole = +2|μ| i` in the upper half plane.  As `|μ|` increases:

1. `xstar² = -μ/ε` scales linearly, so the vertex coefficient in the
   polytope integrand grows.
2. The pole magnitude scales linearly, so `exp(pole · τ)` factors
   inside the mode-sum reach larger magnitudes.
3. Higher-order terms in the polynomial expansion (from products of
   vertex factors during the mode-sum residue closure) carry
   `xstar^(2n)` — those terms grow as a polynomial of degree `2n` in
   `|μ|`.

Together these compound: the symbolic expression tree the integrator
walks gets deeper AND wider, and Sage's `simplify_full` /
`subs` calls on the resulting expressions slow down accordingly.
That is the runtime explosion.

The slowdown is **not** a numerical-precision artifact (no mpmath
fallback fires, no overflow, no NaN), it is purely symbolic-
manipulation cost growing as a polynomial of `|μ|`.

## Why "make the barrier bigger" is the wrong fix

The natural reaction to a hopping sim (`mu = -1` doesn't trap
strongly enough for the chosen `T_sim`, `D`) is to deepen the
wells by making `|μ|` larger.  Kramers barrier height is
`μ²/(4ε)`, and escape time is `exp(barrier / D)`.  So `μ = -3`
gives a much taller barrier than `μ = -1`.

But that increases `|μ|`, which is precisely the knob that breaks
Phase J at ell ≥ 1.  Both the sim and the theory want the SAME
parameter values, so trapping via barrier height conflicts with
theory tractability.

The right trapping knob is `D` (the noise strength).  Lowering `D`
shrinks the noise without touching `μ`, so:

- `barrier / D` grows → escape time grows → sim traps;
- `|μ|` stays moderate → Phase J stays fast.

## Recommendations

For double-well theory ↔ simulation comparison on this theory:

| Knob | Recommended value | Why |
|---|---|---|
| `μ`     | -1.0   | Keeps Phase J fast at ell ≥ 1; non-symmetric saddle. |
| `ε`     | 0.1    | xstar = ±√10 ≈ 3.16, well-separated wells. |
| `D`     | 0.1    | Barrier/D = 25 → escape time ~7×10¹⁰. |
| `T_sim` | 2 × 10⁶ | Easily inside escape time. |
| `max_ell` | 1   | 1-loop comparison practical here. |

Expected ell = 1 Phase J runtime at these params: ~0.3 s for a
3-point τ grid, scaling roughly linearly in the τ-grid size.

If you specifically need `|μ| ≥ 2`:

- Drop to `max_ell = 0` (tree only); tree-level integration is
  ell-independent and finishes instantly at any `|μ|`.
- Or accept multi-hour runs at ell = 1; runtime grows as a
  polynomial of `|μ|` so even `μ = -2` will take notably longer
  than `μ = -1.5`.

## Out-of-scope fixes

A proper code-level fix would attack the symbolic-expansion cost in
the poset modesum integrator — either by switching to a
fast-callable / numerical evaluation path at integrand build time
(avoiding the deep SR expression trees) or by detecting
high-`|μ|` regimes and routing them to scipy.nquad on the summed
integrand (the grouped Phase J path).  Both are significant
engineering efforts and are not attempted in this branch; the
parameter-selection workaround above is the practical answer.

The grouped Phase J path was spot-tested at `μ = -3` and also did
not finish in 10 minutes, so the bottleneck is shared between the
per-diagram and grouped paths — it is in the integrand
construction, not the polytope integration itself.

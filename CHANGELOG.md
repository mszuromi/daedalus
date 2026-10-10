# Changelog

This file starts at 0.2.0. The 0.2.0 entry covers the Phase J changes that
move numbers, and the tools that go with them. Other changes since 0.1.0 are
in the git history.

Entries that change numbers list every public model whose results moved, with
values before and after. "Before" is the code immediately before the change,
which still used the 0.1.0 rules; `DAEDALUS_PHASE_J_LEGACY=1` reproduces it.
"After" is the new default.

## 0.2.0 (unreleased)

### Changed: Itô equal-time rule for constant constraint rows (numbers move)

δ-elimination can leave a smooth (non-instantaneous) propagator whose time
difference is a constant. This happens when instantaneous (δ) parts merge
both of its endpoints, or tie each endpoint to an external time. Its
constraint row is then a constant, and its Heaviside factor is Θ of that
constant, which can be Θ(0).

Daedalus uses the Itô convention Θ(0) = 0. It is the same convention that
evaluates the k = 2 equal-time point as the left limit τ → 0⁻. Every Phase J
integrator already applied it except the two-time (m = 2) polygon integrator.
That integrator kept such a row, which amounts to Θ(0) = 1.

0.2.0 applies one exact rule everywhere, on both the per-diagram and the
grouped Phase J path:

- A constant row whose value is > 0 leaves the region as it is. The
  propagator's exponential factor is kept.
- A constant row whose value is < 0 makes the region empty.
- A constant row whose value is exactly 0:
  - if the value does not depend on the external times (δ parts merge both
    endpoints into one time), Θ(0) = 0: the region is empty. This is the
    change in the polygon integrator, and it moves the numbers below at
    every τ.
  - if the value is the difference of two external times, t_p − t_q, and
    those times coincide, exactly one of Θ(t_p − t_q) and Θ(t_q − t_p)
    holds. The times are compared as the caller gave them; when they are
    exactly equal, the leg later in the argument list counts as
    infinitesimally earlier. The value at coincident external times is
    therefore the one-sided limit in which the later-listed leg approaches
    from below. For k = 2 that is the Itô left limit τ = t₁ − t₀ → 0⁻ that
    the τ grid already samples (at τ = −1e−6). Before 0.2.0 such a pair was
    Θ(0) = 0 on both orientations (Θ(0) = 1 on both in the polygon
    integrator), and the value at coincident times was neither one-sided
    limit. The value at a tie is that limit only as accurately as the tie
    is evaluated: an exact tie can send an integration region to another
    integrator than the points around it, and two of those routes had
    known errors, the `scipy.nquad` fallback (default tolerance; hardened
    by "Changed: hardened quadrature fallback" below) and, for regions
    with three or more integration times, the poset lower-bound
    inheritance error (see the known issue under the k = 2 table below;
    fixed by "Fixed: poset lower-bound inheritance …" below).
    A value at a tie can then be much less accurate than the values around
    it: on the default k = 4 `dd.run` slices of
    `single_population_spike_reset_test`, 1.0% to 8.5% relative (about
    1e−3 absolute) at τ = 0.5, before that fix. See the known issue after
    the k = 3 table below.
  - **So at coincident times of legs of different fields the value depends
    on the order in which `external_fields` lists them.** Where the
    cumulant jumps, listing the same (field, time) pairs in another order
    can give the other one-sided limit. The k = 2 τ = 0 point has worked
    this way since 0.1.0: ⟨A B⟩(τ = 0) is sampled with B just before A, so
    ⟨A B⟩(0) and ⟨B A⟩(0) are the two one-sided limits of the same
    function. For `single_population_linear_delta_spikes_test` (parameters
    in the k = 3 table below) at tree level, ⟨n₁ n₂⟩(0) = +2.53696e−2 and
    ⟨n₂ n₁⟩(0) = +2.39076e−2, before and after 0.2.0. At k = 3, the
    configuration n₁ at 0, n₂ at 0.7 and n₁ at 0.7 gives +1.56695e−4 (n₁
    just before n₂) when the fields are listed as (n₁, n₂, n₁) at
    (0, 0.7, 0.7), and +6.34919e−4 (n₂ just before n₁) when they are listed
    as (n₁, n₁, n₂) at (0, 0.7, 0.7). Before 0.2.0 both orders gave
    +8.07043e−5, neither limit. Legs of the same field give the same value
    in either order, and so do distinct times.
- There is no tolerance: external times that differ, however slightly, are
  never treated as coincident, and no external time is ever modified. A
  value that differs from 0 only by rounding (for example
  `0.1 + 0.2 - 0.3`) is decided by its sign, as before 0.2.0.
- A region that is empty by construction evaluates to exactly 0, without
  quadrature. Each of these tests is exact, with no tolerance either:
  - a two-time region bounded by two rows with exactly opposite normals whose
    constants sum to ≤ 0 (for example s₁ > s₀ and s₀ > s₁, or a strip
    between two exactly coincident external times), or a clipped two-time
    region of exactly zero area;
  - a directed cycle of ordering rows whose shifts sum to ≤ 0;
  - a δ-subset with an empty row that does not depend on τ. The subset is
    still built; the mode is checked when it is evaluated, so changing the
    Θ(0) flag after `compute_cumulants` returns takes effect.

  A genuinely thin region is integrated as before 0.2.0, for example the
  strip between external times 1e−12 apart.

  Before 0.2.0, many of these regions went to the `scipy.nquad` fallback,
  which already applied Θ(0) = 0. The affected runs now make far fewer
  quadrature calls; see the tables below.

**Which evaluations are affected.** Constant rows come from δ-elimination,
so only models whose propagators have instantaneous (δ) parts are affected,
and only where a constant row is exactly 0:

- a row that does not depend on the external times: at every τ (the
  spike-reset k = 1 and k = 2 tables below);
- a row comparing two external times: only where those times coincide
  exactly. The k = 2 τ grid never evaluates coincident times (τ = 0 is
  sampled at −1e−6, and `total_C(t, t)` is nudged the same way), so k = 2
  curves move only through the first case. In the k ≥ 3 curves of `dd.run`
  (`C_tau` and `C_tau_slices` of a k ≥ 3 result) the swept leg never
  coincides with another leg either (next section). But with
  `kpoint_base_lags` left at its default, every non-swept non-anchor leg
  sits at −1e−6, so:
  - for k ≥ 4, EVERY point of a slice has at least two coincident legs (the
    k − 2 non-swept ones), so a whole slice curve can move, and so can a
    k ≥ 4 moment output (`Config.output = 'moment'` or `'central_moment'`,
    evaluated at the times of slice 1, next section);
  - the full grid (`kpoint_full_grid`, `C_tau_grid`) has coincident legs on
    its diagonals.

  A point with coincident legs moves only where the two one-sided limits
  differ, for example legs of different fields of a model with
  instantaneous parts (tables below). With distinct, nonzero
  `kpoint_base_lags` the non-swept legs stay apart.

Models without instantaneous parts (in every re-run: the OU family and the
spatial models) have no constant rows. Their numbers are unchanged at every
point, including coincident and nearly coincident external times. In
general, the new rules can change a value only where the evaluation decides
a constant row, skips a τ-independent empty subset, or applies one of the
exact emptiness tests above. When the counters `theta0_const_empty`,
`theta0_const_drop`, `theta0_subsets_pruned`, `polygon_zero_area` and
`poset_empty_cycle` all stay 0, the value is identical to the old one.
(`zero_normal_rows_seen` alone is not enough: it counts constant rows only
in the two-time and multi-time integrators.) The τ = 0 points of the
`dd.run` k ≥ 3 slices of these models, and the points of their k ≥ 3
moment outputs, do shift slightly, because the times at which those points
are evaluated change (next section).

### Changed: `dd.run` k ≥ 3 slices take the left limit where legs meet (numbers move)

`dd.run` turns a k ≥ 3 cumulant into k − 1 curves over `tau_grid`
(`C_tau_slices`; `C_tau` is slice 1). Slice j sweeps leg j
(τ_j = t_j − t₀), and the other non-anchor legs stay at their
`kpoint_base_lags` (default 0, evaluated at −1e−6 like the k = 2 grid's
τ = 0 point). Before 0.2.0 every non-anchor leg with |t| ≤ 1e−12 was moved
to −1e−6, so at τ = 0 the swept leg landed on the same time as the other
non-anchor legs at a base lag of 0. The value of that point was then set by
the pre-0.2.0 Θ(0) handling of the coincident legs, not by the curve, and
where the curve jumps at τ = 0 (legs of different fields in a model with
instantaneous parts) it was neither of the curve's one-sided limits. The
tie order of the previous section alone would not make it the left limit:
it takes the limit in which the later-listed leg approaches from below,
which is the curve's left limit only on a slice whose swept leg is the
last-listed of the coincident legs.

In 0.2.0 every point of a slice is a point of its own curve:

- Where the swept leg meets another leg (within 1e−12: the anchor at τ = 0,
  or a non-swept leg at τ equal to its base lag or to its evaluation time;
  for a base lag of 0 that is τ = 0 or τ = −1e−6), it is placed 1e−6
  (`api.compute._ITO_EPS`) below the earliest leg it meets, again if that
  lands on another leg (where 1e−6 is below the float spacing of the time,
  |τ| ≥ 2³⁴ ≈ 1.7e10, the next float below). The point is then the left
  limit τ → τ₀⁻ of that slice, as the k = 2 grid's τ = 0 point is. With the
  default base lags, the τ = 0 point of a slice has the swept leg at −2e−6
  and the other non-anchor legs at −1e−6; a grid point at τ = −1e−6 gets
  the same times, and the same value, as the τ = 0 point.
- Every other point is evaluated at exactly the same times as before, and
  its value is unchanged (bit-for-bit, in process).
- The point is evaluated 1e−6 below the leg it meets, not in the limit, so
  it equals the left limit at τ₀ only up to the curve's change over that
  offset: 2e−6 at τ = 0 with the default base lags (1e−6 for the k = 2
  grid's τ = 0 point). For rates of order 1 that is of order 1e−6
  relative. It does not matter where the curve jumps by much more
  (`single_population_spike_reset_test` and
  `single_population_linear_delta_spikes_test` below), but it does where
  the jump is itself that small: the τ = 0 points of `multipopulation_test`
  are 1.8e−6 and 2.4e−6 relative from the left limits at τ = 0 (slices 1
  and 2), while its jumps there are 4.6e−6 and 5.6e−7. On a slice whose
  swept leg is the last-listed of the legs it meets (slice 2 at k = 3,
  slice 3 at k = 4), the tie order alone, at the old point
  (0, −1e−6, …, −1e−6), would already give the left branch, 1e−6 left of
  τ = 0; the new point, 2e−6 left of τ = 0, is about twice as far from the
  left limit at τ = 0: 6.0e−5 relative instead of 3.0e−5
  (`single_population_spike_reset_test` k = 4 slice 3) and 2.0e−7 instead
  of 9.8e−8 (`single_population_linear_delta_spikes_test` k = 3 slice 2).
- τ-grid points closer than about 2e−6 to 0 or to a base lag are below the
  resolution of this placement. A τ strictly between a non-swept leg's
  evaluation time and its base lag (for a base lag of 0: −1e−6 < τ < 0) is
  evaluated as given, with the swept leg between that leg and the anchor,
  while τ = −1e−6 and τ = 0 are left limits. Base lags within a few 1e−6
  of 0 or of each other are below the resolution too.
- The non-swept legs are never moved. Where they coincide with each other
  (k ≥ 4 with the default base lags, where they all sit at −1e−6), the tie
  order of the previous section decides, the same way at every point of
  the slice, so it adds no jump to the curve. But then every point of the
  slice is evaluated at a tie, and only as accurately as the tie is (known
  issue after the k = 3 table below). For
  `single_population_spike_reset_test` k = 4 (tree level, parameters of the
  tables below) the tie sends regions with three integration times to the
  poset integrator, where the known lower-bound inheritance error applies:
  at τ = 0.5 the points of slices 1, 2 and 3 are 2.1%, 1.0% and 8.5%
  relative (9.9e−4, 4.9e−4 and 9.9e−4 absolute) off the tie-order limit.
  The τ = 0 points and right limits tabulated below are not affected: the
  τ = 0 points of slices 1, 2 and 3 equal the tie-order limit to ≤ 2.3e−11
  relative, and the points at τ = 2e−5 (slices 1 and 3) to 5.7e−11 and
  1.8e−10. Distinct, nonzero `kpoint_base_lags` keep the non-swept legs
  apart.
- `result['_kpoint_slice_times']` lists the external times of every slice
  point.
- Unchanged: k = 2 results, the full grid (`kpoint_full_grid`,
  `C_tau_grid`: each leg is mapped on its own, and legs with equal values
  stay coincident on its diagonals), and the callables `total_C` and
  `total_C_by_ell`. So the full grid keeps the tie order at its coincident
  points (its origin and diagonals), and there it can differ from the
  slices: for `single_population_linear_delta_spikes_test` k = 3 (tree,
  parameters of the k = 3 table below), `C_tau_grid` at the origin is
  +6.80526e−4, the τ = 0 point of slice 2 (to 9.8e−8 relative), while the
  τ = 0 point of slice 1 is +6.60665e−4.
- `DAEDALUS_PHASE_J_LEGACY=1` restores the Phase J rules, not the slice
  times. With it set, the pre-0.2.0 τ = 0 point is
  `result['total_C'](0, -1e-6, ..., -1e-6)`.
- The k ≥ 3 moment outputs (`Config.output = 'moment'` or
  `'central_moment'`) follow slice 1. They are the curve
  ⟨φ(0) φ(τ) φ(0) … φ(0)⟩ over `tau_grid` (leg 1 swept, the other legs at
  0; `kpoint_base_lags` is not used, as before). Their cumulant blocks of
  3 or more legs used to be evaluated with every leg but the swept one at
  exactly 0, the anchor's time; they are now evaluated at the times of
  slice 1 with the default base lags. So the k = 3 central moment equals
  `C_tau` bit-for-bit, and its τ = 0 point is the left limit too: for
  `single_population_linear_delta_spikes_test` ⟨n₁ n₂ n₁⟩ (tree,
  parameters of the k = 3 table below) it moves from +6.80526e−4, the
  tie-order value at (0, 0, 0), to +6.60665e−4, 2.9% lower (the raw moment
  by 1.7e−5 relative). Its other points move by the curve's change over
  the 1e−6 offset of the non-swept leg: by ≤ 1.2e−7 relative at τ = ±0.5
  and for 3e−6 ≤ |τ| ≤ 1e−4. For models without instantaneous parts the
  shift is of the same kind: `ou_quartic_double_well` k = 3 (`mu = -1`,
  `eps = 0.1`, `D = 0.1`, tree) by 7.7e−7 relative at τ = ±0.5 and
  ≤ 4.1e−10 for |τ| ≤ 1e−4, and at its default parameters k = 4 (tree plus
  one loop) by 1.2e−7 at τ = ±0.5 and ≤ 6.6e−12 for |τ| ≤ 1e−4. At k ≥ 4
  the non-swept legs coincide with each other, as on the slices, so every
  point of a k ≥ 4 moment output is evaluated at a tie (known issue after
  the k = 3 table below). Blocks of 2 legs read the k = 2 grid as before.

τ = 0 point of every slice, default base lags, tree level. The legs
alternate between two fields: ⟨n₁ n₂ n₁⟩ and ⟨n₁ n₂ n₁ n₂⟩
(`multipopulation_test`: n_E₁, n_E₂; `dendritic_quad_soma_sigmoid`: n_S₁,
n_S₂), with the parameters of the tables below (`dendritic_quad_soma_sigmoid`:
its default parameters). "0.1.0" is the old point (0, −1e−6, …, −1e−6) under the 0.1.0 rules,
"0.2.0" the new point. The new point lies on the left branch of its slice:
it equals the Lagrange extrapolation of the slice from τ = −1e−4, −1e−5,
−3e−6 to the swept leg's time (−2e−6) to ≤ 1.3e−13 relative (≤ 2.1e−11 for
`single_population_spike_reset_test`, whose k ≥ 3 tree level is partly
evaluated by `scipy.nquad`); the left limit at τ = 0 itself differs from it
by the offset described above. "right limit" is the limit τ → 0⁺ of the
same slice (from τ = 1e−4, 1e−5, 3e−6).

| model | k | slice | 0.1.0 | 0.2.0 | right limit |
|---|---|---|---|---|---|
| `single_population_spike_reset_test` | 3 | 1 | +1.01360e−1 | −3.32555e−1 | −2.34321e−1 |
| | 3 | 2 | +1.01360e−1 | +4.96369e−2 | −3.32555e−1 |
| | 4 | 1 | −1.85754e−2 | +5.96347e−2 | −7.58513e−2 |
| | 4 | 2 | −1.85754e−2 | −5.29033e−2 | +5.96348e−2 |
| | 4 | 3 | −1.85754e−2 | +2.12106e−3 | +2.48028e−2 |
| `single_population_linear_delta_spikes_test` | 3 | 1 | +2.64260e−5 | +6.60665e−4 | +1.55065e−4 |
| | 3 | 2 | +2.64260e−5 | +6.80526e−4 | +6.60666e−4 |
| | 4 | 1 | +1.23614e−6 | +7.25644e−6 | +3.15438e−5 |
| | 4 | 2 | +1.23614e−6 | +2.99907e−5 | +7.25644e−6 |
| | 4 | 3 | +1.23614e−6 | +3.27274e−5 | +3.12428e−5 |
| `multipopulation_test` | 3 | 1 | +9.6992059e−7 | +9.6992026e−7 | +9.6992289e−7 |
| | 3 | 2 | +9.6992059e−7 | +9.6992218e−7 | +9.6991930e−7 |
| `dendritic_quad_soma_sigmoid` | 3 | 1 | +3.18218e−6 | +1.31101e−5 | +1.87098e−5 |
| | 3 | 2 | +3.18218e−6 | +1.58915e−5 | +1.31101e−5 |

- Where the two one-sided limits of a slice differ (legs of different
  fields meeting in a model with instantaneous parts), the τ = 0 point is
  now the left one. Before 0.2.0 it was neither.
- Where the curve is continuous, the point moves only because the swept
  leg moves by 1e−6: `linear_hawkes` (parameters `P_LINH` of
  `tests/tools/phase_j_zoo_baseline.py`) by 4.4e−7 and 4.8e−7 relative at
  k = 3 (slices 1 and 2; its one-loop term is 0 there) and by ≤ 5.5e−7 at
  k = 4; `single_population_quad_exp_test` (`P_SP`) by 5.5e−7 and 1.1e−6
  at k = 3. `quadratic_hawkes_alpha` (default parameters, ⟨n₁ n₂ n₁⟩) has
  instantaneous parts, but its k = 3 slices are continuous at τ = 0 to
  within their evaluation accuracy (the one-sided limits are 2.3e−9 and
  3.0e−10 relative apart): its points move by 1.3e−8 and 5.6e−9 relative
  and equal the left branch to 2.1e−10 and 1.1e−10. For the models without
  instantaneous parts that were
  measured, the curves are flat to first order at the default τ = 0 point
  and the shift is O(1e−12): `ou_quartic` (`mu = 1`, `eps = 0.02`, `D = 1`)
  k = 4 by 2.5e−12 (tree) and 1.1e−12 (one loop) relative (its k = 3
  cumulant is identically 0, before and after); `ou_quartic_double_well`
  at `mu = -1`, `eps = 0.1`, `D = 0.1` (a nonzero saddle, so k = 3 is not
  0) k = 3 by 8.0e−12 (tree), and at its default parameters (`mu = 1`, a
  symmetric saddle) k = 4 by 2.5e−12. The other public models without
  instantaneous parts, at their default parameters, tree level, every
  slice: `ou_quartic_colored` k = 4 by 5.3e−11, `ou_quartic_two_dim_color_corr`
  (⟨x y x y⟩) k = 4 by 4.9e−11, and `ou_sextic` and
  `toy_quartic_double_well` k = 4 by 2.5e−12. Their k = 3 cumulants
  (⟨x y x⟩ for `ou_quartic_two_dim_color_corr`) are 0, before and after
  (below 1e−300 in absolute value).
- With nonzero base lags the crossing points are left limits too. For
  example, for `single_population_linear_delta_spikes_test` k = 3 with
  `kpoint_base_lags=[0.5, -1.0]`, the point of slice 1 at τ = −1 (the swept
  n₂ leg meets the n₁ leg at −1) is +6.12238e−4, the left limit; the
  coincident point (0, −1, −1) itself gives +6.28418e−4 with the tie order.
  On slice 2 the swept leg was already the earlier one at its crossing
  (τ = 0.5), which moves by 1.1e−7 relative. A crossing point of a model
  without instantaneous parts moves by O(1e−6) relative (`ou_quartic` k = 4
  with `kpoint_base_lags=[0.5, 0.5, 0]`: ≤ 5.0e−7 per loop order, and
  6.5e−7 for the total, tree plus one loop).
- Not measured (each over its time budget): `multipopulation_test` and
  `single_population_quad_exp_test` at k = 4 (builds not finished within
  90 and 45 minutes); `dendritic_quad_soma_sigmoid` and
  `quadratic_hawkes_alpha` at k = 4 (builds not finished within 30
  minutes); `ou_quartic_double_well` at `mu = -1`, `eps = 0.1`, `D = 0.1`
  at k = 4 (one tree evaluation not finished within 30 minutes). The first
  four have instantaneous parts, so the τ = 0 points of their k = 4 slices
  can jump as those of `single_population_spike_reset_test` do; the last
  has none, so its points move by O(1e−6) relative or less.
- No tracked notebook evaluates a k ≥ 3 `dd.run` slice or moment output.

### Moved (measured)

`single_population_spike_reset_test`, parameters `Em = [3.5, 3.5]`,
`tau = [10, 9]`, `a = [2.5, 2.5]`, `w = [[0.55, 0.65], [0.7, 0.8]]`.

These are the package's current values, not validated results: the model
has no exact solution, and it serves in the tests to exercise the code
paths this release changes (instantaneous propagator parts, the
fallback integrator, coincident times), each region being checked against
an independent high-precision value of the same integral. The reliable
public accuracy references are the Ornstein–Uhlenbeck family (exact
stationary densities) and the linear point-process models, whose loop
corrections vanish exactly.

k = 2, `max_ell = 1`, ⟨n₁ n₂⟩(τ). The tree level (ℓ = 0) is unchanged.

| τ | one-loop term, before | one-loop term, after | total, before | total, after |
|---|---|---|---|---|
| 0 | −4.25258e−2 | −5.06015e−3 | +4.88463e−1 | +5.25929e−1 |
| 1 | −3.41721e−2 | −6.36836e−3 | +1.39192e−1 | +1.66995e−1 |
| 3 | −4.61861e−3 | −1.29531e−3 | −2.11620e−3 | **+1.20710e−3** (sign flip) |
| 5 | +1.22483e−3 | +1.19179e−4 | −3.54957e−3 | −4.65523e−3 |
| 10 | +2.48020e−4 | +4.65705e−5 | −7.06296e−5 | −2.72080e−4 |

- The largest change in the total is 3.75e−2 absolute, at τ = 0. The largest
  relative change is 285%, at τ = 10.
- **Known issue (fixed by "Fixed: poset lower-bound inheritance …"
  below): the "after" one-loop values of this table were not converged.**
  They contained a separate error of the multi-time integrator (three or
  more integration times): in a causal ordering, a time variable could
  inherit a lower integration bound it should not have (poset lower-bound
  inheritance). At these parameters the error is large away from τ = 0.
  Measured against tight quadrature on every affected integration region
  (96 of the 520 multi-time regions), the one-loop term should be
  −3.866e−3 at τ = 1 (not −6.368e−3, an error of +2.50e−3) and −1.026e−3
  at τ = 3 (not −1.295e−3, +2.69e−4; the total there is then about
  +1.48e−3, so the sign flip stands). At the τ = 0 grid point the error is
  below 1e−8 absolute. That fix moves the one-loop term to these values
  (−3.86596e−3 and −1.02615e−3) and also at τ = 5 and 10; its table below
  gives them. The frozen fixture stays a strict expected failure until it
  is refrozen.
- Per-diagram and grouped Phase J give the same new values, to a relative
  2e−11 to 7e−9.
- `scipy.nquad` fallback calls drop from 360 to 40 (per-diagram) and, for
  grouped Phase J, from 180 to 10 with a warm developer cache (30 to 10 from
  a fresh one; the counts depend on the diagram cache).

k = 1, `max_ell = 2`, stationary mean ⟨n₁⟩. The loop corrections at the
mean-field saddle:

| term | before | after |
|---|---|---|
| one-loop | −3.79871e−2 | −3.79871e−2 (unchanged) |
| two-loop | +3.03944e−2 | **−8.89854e−4** (sign flip) |

The `scipy.nquad` fallback calls drop from 464 to 0. With `max_ell = 1`, k = 1
is unchanged.

k = 3 at coincident external times, tree level (the one-loop terms of linear
delta spikes and multipopulation are exactly 0 here), ⟨n₁(t₀) n₂(t₁) n₁(t₂)⟩,
spike-reset parameters as above. The
"limit" column is the one-sided limit in which the later-listed of the
coincident legs approaches from below; "other limit" is the other one-sided
limit, given where it differs (Richardson extrapolation from
h = 1e−4, 1e−5, 1e−6). At (0, 0, 0) the three legs have three distinct
limits: n₂ between the two n₁ legs (the tie order), n₂ first and n₂ last.
The rows marked `dd.run` are the τ = 0 points of the k = 3 slices (see
"Changed: `dd.run` k ≥ 3 slices …" above): "before" is the old point
(0, −1e−6, −1e−6), where the
swept leg coincided with the other one; "after" is the new point, with the
swept leg at −2e−6; "limit" is the left branch of that slice at the swept
leg's time −2e−6 (Lagrange extrapolation from τ = −1e−4, −1e−5, −3e−6),
and "other limit" is the right limit of the slice at τ = 0 (from τ = 1e−4,
1e−5, 3e−6). The left limit at τ = 0 itself differs from "limit" by the
curve's change over 2e−6 (next bullets).

| model | (t₀, t₁, t₂) | before | after | limit | other limit |
|---|---|---|---|---|---|
| `single_population_spike_reset_test` | (0, 0.7, 0.7) | −4.58858e−4 | −9.89874e−2 | −9.89874e−2 | +4.12483e−2 |
| | (0.3, 0.3, 0.7) | +1.02854e−1 | −2.13653e−1 | −2.13653e−1 | +4.09230e−2 |
| | `dd.run` slice 1, τ = 0: (0, −2e−6, −1e−6) | +1.01360e−1 | −3.32555e−1 | −3.32555e−1 | −2.34321e−1 |
| | `dd.run` slice 2, τ = 0: (0, −1e−6, −2e−6) | +1.01360e−1 | +4.96369e−2 | +4.96369e−2 | −3.32555e−1 |
| | (0, 0.4, 0) | +2.25093e−2 | −1.43972e−1 | −1.43972e−1 | (same) |
| `single_population_linear_delta_spikes_test`, `Em = [0.8, 0.78]`, `tau = [10, 9]`, `w = [[0, 0.25], [0.2, 0]]` | (0, 0.7, 0.7) | +8.07043e−5 | +1.56695e−4 | +1.56695e−4 | +6.34919e−4 |
| | (0.3, 0.3, 0.7) | +2.56166e−5 | +6.40873e−4 | +6.40873e−4 | +6.59190e−4 |
| | `dd.run` slice 1, τ = 0: (0, −2e−6, −1e−6) | +2.64260e−5 | +6.60665e−4 | +6.60665e−4 | +1.55065e−4 |
| | `dd.run` slice 2, τ = 0: (0, −1e−6, −2e−6) | +2.64260e−5 | +6.80526e−4 | +6.80526e−4 | +6.60666e−4 |
| | (0, 0.4, 0) | +1.48826e−4 | +1.48826e−4 | +1.48826e−4 | (same) |
| | (0, 0, 0) | +1.06073e−5 | +6.80526e−4 | +6.80526e−4 (n₂ between) | +6.60666e−4 (n₂ first), +1.55065e−4 (n₂ last) |
| `multipopulation_test` (parameters of `tests/tools/phase_j_zoo_baseline.py`, `P_MP`), ⟨n_E₁ n_E₂ n_E₁⟩ | (0, 0.7, 0.7) | +2.4548779e−6 | +2.4548792e−6 | +2.4548792e−6 | +2.4548736e−6 |
| | (0.3, 0.3, 0.7) | +1.3250094e−6 | +1.3250003e−6 | +1.3250003e−6 | +1.3250139e−6 |
| | `dd.run` slice 1, τ = 0: (0, −2e−6, −1e−6) | +9.6992059e−7 | +9.6992026e−7 | +9.6992026e−7 | +9.6992289e−7 |
| | `dd.run` slice 2, τ = 0: (0, −1e−6, −2e−6) | +9.6992059e−7 | +9.6992218e−7 | +9.6992218e−7 | +9.6991930e−7 |
| | (0, 0, 0) | +9.7179215e−7 | +9.6992006e−7 | +9.6992006e−7 (n_E₂ between) | +9.6991841e−7 (n_E₂ first), +9.6992172e−7 (n_E₂ last) |

- The new values equal the limit to ≤ 1e−15 relative (linear delta spikes),
  ≤ 6e−14 (multipopulation) and ≤ 7e−11 (spike reset, whose k = 3 tree
  level is partly evaluated by `scipy.nquad`). The `dd.run` rows equal
  their "limit" (the left branch at −2e−6) to ≤ 1.2e−15, ≤ 1.3e−13 and
  ≤ 5.6e−13, and differ from the left limit at τ = 0 itself by 3.8e−7 and
  2.0e−7 (linear delta spikes, slices 1 and 2), 1.8e−6 and 2.4e−6
  (multipopulation) and 2.6e−6 and 9.4e−7 (spike reset) relative.
- The one-sided limits of `multipopulation_test` at the `dd.run` τ = 0
  points differ by only 4.6e−6 (slice 1) and 5.6e−7 (slice 2) relative,
  while its curves change by about 1e−6 relative per 1e−6 of τ. Its
  `dd.run` points, 2e−6 left of τ = 0, are therefore left limits only to
  that level: slice 2's point is further from its left limit at τ = 0 than
  the jump itself.
- The same k = 3 evaluations at distinct times, including times 1 ulp,
  1e−13, 1e−12 and 3e−12 apart and times shifted by 10⁶, are unchanged
  (bit-for-bit, in process). So is the k = 4 point (0, 0.3, 0.3 + 1e−12, 0.9)
  of linear delta spikes.
- At coincident times, `linear_hawkes` k = 3 (ℓ ≤ 1) values change by
  ≤ 3.1e−14 relative and `single_population_quad_exp_test` k = 3 tree values
  by ≤ 1.7e−14.
- **Known issue: a value at exactly coincident times is the one-sided limit
  only as accurately as it is evaluated.** An exact tie can send an
  integration region to another integrator than the points around it, and
  two of those routes had known errors:
  - Two-time regions can go to the `scipy.nquad` fallback and carried its
    error. For `multipopulation_test` ⟨n_E₁ n_E₂ n_E₁⟩ at tree level
    (parameters as above), with a fresh diagram cache, (0, 0.4, 0.4) sends
    16 such regions there (none at (0, 0.4, 0.4 − 1e−7) or at
    (0, 0.7, 0.7); with other diagram representatives all of these points
    can reach it, see "Changed: hardened quadrature fallback" below). With
    the default-tolerance
    fallback its value, +1.68597e−6, was 2.9e−5 relative from the limit,
    +1.68592e−6: twelve times the jump between the two one-sided limits
    there (2.4e−6), so it was neither limit. With the hardened fallback
    ("Changed: hardened quadrature fallback" below) it is
    +1.68592416791e−6, the limit to 5.9e−15 relative. The ties (0, t, t)
    of the same model at t = −2, −1, −0.5, 0.2, 0.5, 0.7, 1 and 2 reach no
    fallback and equal the limit to ≤ 1.4e−14.
  - At an exact tie, a region with three or more integration times can be
    accepted by the analytic poset integrator, which rejects it (and sends
    it to the fallback) when the tied times differ by more than 1e−9.
    Accepted, it carried the poset lower-bound inheritance error (the
    known issue under the k = 2 table above): two tied external lower
    bounds count as consistent, and a time variable that has no lower
    bound of its own inherited theirs. The error is large. "Fixed: poset
    lower-bound inheritance …" below refuses such a region and integrates
    it exactly; the k = 4 values in this item were measured before that
    fix and have not been re-measured after it. For
    `single_population_spike_reset_test` ⟨n₁ n₂ n₁ n₂⟩ at tree level
    (parameters as above), every point of the default k = 4 `dd.run`
    slices has its two non-swept legs tied at −1e−6. At τ = 0.5 the values
    are −4.82409e−2, +4.73977e−2 and +1.07140e−2 (slices 1, 2 and 3),
    while the tie-order limits are −4.72516e−2, +4.78860e−2 and
    +1.17033e−2: 2.1%, 1.0% and 8.5% relative, 9.9e−4, 4.9e−4 and 9.9e−4
    absolute. (The limits are extrapolated from the tie-order leg moved
    2e−7, 1e−7 and 5e−8 lower. Those points send the regions to the
    fallback instead: 1768 fallback entries per point, against 1544 to
    1568 at the tie.) Slice 2's tied legs belong to the same field, so
    both orientations give the same limit, and the tie value is off it
    too. The τ = 0 points of the slice table above are not affected (they
    equal their limit to ≤ 2.3e−11 relative), and neither are the right
    limits there (at τ = 2e−5, 5.7e−11 and 1.8e−10 for slices 1 and 3).
    On a hand-built region with this structure the value is too small by
    a factor of about 5e5 at an exact tie, and also when the two times
    differ by 1e−10.
  - The ties of the k = 4 slices of
    `single_population_linear_delta_spikes_test` (τ = 0 and 0.5) and of
    `linear_hawkes` (τ = 0.5) reach no fallback and equal the tie-order
    limit to ≤ 7.4e−15.

  Such points come from `total_C` at coincident times and, in `dd.run`,
  from the diagonals of the full grid, the coincident non-swept legs of
  k ≥ 4 slices (with the default base lags, every point of a k ≥ 4 slice)
  and every point of a k ≥ 4 moment output (`Config.output = 'moment'` or
  `'central_moment'`, whose cumulant blocks of 3 or more legs are
  evaluated at the times of slice 1). The fallback is hardened by
  "Changed: hardened quadrature fallback" below, and the poset
  lower-bound error by "Fixed: poset lower-bound inheritance …" below
  (the hardening moves the k = 4 slice-1 value at τ = 0.5 above by
  1.2e−8 relative; the inheritance fix was not re-measured there).

k = 4, tree level, ⟨n₁(t₀) n₂(t₁) n₁(t₂) n₂(t₃)⟩ of
`single_population_linear_delta_spikes_test` (parameters as above), at the
points of the default `dd.run` k = 4 slices: the swept leg at τ, the other
non-anchor legs at −1e−6. "limit" is the one-sided limit in which the
later-listed of the coincident legs approaches from below, "other limit"
the other one (Richardson extrapolation from h = 1e−7, 1e−8, 1e−9). The
τ = 0 rows are the points of the slices at τ = 0, where the swept leg is
now at −2e−6 (see "Changed: `dd.run` k ≥ 3 slices …" above): "before" is
the old point
(0, −1e−6, −1e−6, −1e−6), "limit" is the slice's left branch at the swept
leg's time −2e−6 (Lagrange extrapolation from τ = −1e−4, −1e−5, −3e−6),
and "other limit" is its right limit at τ = 0 (from τ = 1e−4, 1e−5,
3e−6). The last row is that old point itself, which
`total_C` still evaluates with the tie order (leg 2 between legs 3 and 1);
its other limits are those with leg 2 first and last.

| swept leg | (t₀, t₁, t₂, t₃) | before | after | limit | other limit |
|---|---|---|---|---|---|
| 1 | (0, 0.5, −1e−6, −1e−6) | +1.79275e−6 | +3.00275e−5 | +3.00275e−5 | +2.97737e−5 |
| 1 | (0, 1, −1e−6, −1e−6) | +1.71067e−6 | +2.85880e−5 | +2.85880e−5 | +2.83773e−5 |
| 1 | (0, 2, −1e−6, −1e−6) | +1.55813e−6 | +2.59229e−5 | +2.59229e−5 | +2.57873e−5 |
| 3 | (0, −1e−6, −1e−6, 0.5) | +1.79275e−6 | +2.97737e−5 | +2.97737e−5 | +3.00275e−5 |
| 2 | (0, −1e−6, 0.5, −1e−6) | +6.96331e−6 | +6.96331e−6 (≤ 1.5e−15 relative change) | (same) | (same) |
| 1, τ = 0 | (0, −2e−6, −1e−6, −1e−6) | +1.23614e−6 | +7.25644e−6 | +7.25644e−6 | +3.15438e−5 |
| 2, τ = 0 | (0, −1e−6, −2e−6, −1e−6) | +1.23614e−6 | +2.99907e−5 | +2.99907e−5 | +7.25644e−6 |
| 3, τ = 0 | (0, −1e−6, −1e−6, −2e−6) | +1.23614e−6 | +3.27274e−5 | +3.27274e−5 | +3.12428e−5 |
| (`total_C` only) | (0, −1e−6, −1e−6, −1e−6) | +1.23614e−6 | +3.27274e−5 | +3.27274e−5 | +2.99907e−5 (leg 2 first), +7.25644e−6 (leg 2 last) |

- Every point of the slices whose non-swept legs belong to different fields
  (swept leg 1 or 3) moves, by a factor of about 17. Slices 1 and 3, equal
  before, now take the two different one-sided limits. Slice 2, whose
  non-swept legs belong to the same field, changes only by rounding
  (≤ 1.5e−15 relative) away from τ = 0.
- The new values equal the limit to ≤ 7e−16 relative. The τ = 0 rows equal
  their "limit" (the left branch at −2e−6) to ≤ 3e−15, and differ from the
  left limit at τ = 0 itself by 4.0e−9, 4.1e−7 and 2.7e−7 relative
  (slices 1, 2 and 3).

### Unchanged (measured)

Each of these was compared with the old code side by side in one process,
at the same points, and was bit-for-bit identical. The k ≥ 3 points include
exactly coincident times, times 1 ulp and 1e−13 apart (at k = 3 also
1e−12 and 3e−12 apart) and times shifted by 10⁶. The
pre-change baseline re-runs agree too (largest relative difference ≤ 3e−16,
cross-process rounding); `ou_quartic` k = 2 with ℓ = 3 was checked that way
only.

- `ou_quartic`: k = 2 with ℓ ≤ 3; k = 4 with ℓ ≤ 2 (this is the OU
  evidence at coincident and nearly coincident times). Also k = 3 with
  ℓ ≤ 2, but that cumulant is identically 0 (odd cumulants vanish at the
  symmetric saddle, and no Phase J integral is evaluated), so it says
  nothing about ties.
- `ou_quartic_colored` and `ou_quartic_two_dim_color_corr`: k = 2, ℓ = 1.
- `single_population_quad_exp_test`: k = 2, ℓ = 1. It has constant rows, but
  none of them is zero-valued at the τ grid.
- `linear_hawkes`, `multipopulation_test` and
  `single_population_linear_delta_spikes_test`: k = 2, ℓ = 1 (linear delta
  spikes also ℓ = 2). At k = 3 (ℓ ≤ 1) their values at distinct times are
  unchanged; at coincident times see the k = 3 table above.
- `single_population_quad_exp_test`: k = 3 tree level at distinct times.
- `single_population_spike_reset_test`: tree level (k = 2, ℓ = 0), k = 1
  with ℓ = 1, and k = 3 at distinct times.
- Spatial:
  - `allen_cahn_1d_subcritical_infinite`, ℓ = 2;
  - `reaction_diffusion_2d`, `coupled_rd_2species_1d` and
    `reaction_diffusion_conserved_1d`, ℓ = 1;
  - `edwards_wilkinson_1d`, ℓ = 0.

### May move: not yet re-measured

These public models exceeded the time budget before and after the change, so
they have no before/after comparison yet. Recompute any saved results for them.

- `dendritic_quad_soma_sigmoid`: k = 2, ℓ = 1 (over 15 min).
- `quadratic_hawkes_alpha`: k = 2, ℓ = 1 (over 15 min).
- `reaction_diffusion_2d` and `coupled_rd_2species_1d`: ℓ = 2 (over 10 min).
  These spatial models have no instantaneous parts and are not expected to
  move, but they were not re-run.

Any k ≥ 3 result of a model with instantaneous parts, evaluated at exactly
coincident external times, moves as described above, and so does every
point of its `dd.run` k ≥ 3 slices where the swept leg meets another leg
(τ = 0, a base-lag crossing, or τ = −1e−6 with a base lag of 0). The
k ≥ 3 moment outputs of every model move as described in "Changed:
`dd.run` k ≥ 3 slices …" above.
Models outside the public model set are not listed.

### Performance: spatial runs use the compiled enumerator

The diagram stage of every spatial run, which never uses the prediagram
cache, now takes its prediagrams from the compiled streaming enumerator,
rebuilt in memory, instead of the pure-Python eager enumerator. No file is
read or written. The certificates of each (k, ℓ) are kept in memory for the
rest of the process; the records are rebuilt on every call.

Temporal `compute_cumulants(use_cache=False)` runs keep the eager
enumerator and its exact records, so their numbers and their
`result['diagrams']` do not change (see "Why temporal runs keep the eager
records" below). `DAEDALUS_PREDIAGRAM_EAGER` sets the source for every
caller: `0` (or `false`, `no`, `off`) streams temporal runs too, `1` (or
`true`, `yes`, `on`) puts spatial runs back on the eager enumerator, and any
other non-empty value is an error.

With the defaults, no run is measurably faster yet: temporal runs keep the
eager enumerator, and the spatial integrators stop at ℓ ≤ 2, where the
enumeration is a small part of the run (measured:
`allen_cahn_1d_subcritical_infinite` at ℓ ≤ 2 takes 5.4 s with either
enumerator, of which loading the prediagrams takes 0.10 to 0.13 s). The
temporal gain below needs `DAEDALUS_PREDIAGRAM_EAGER=0`.

- Speed (measured). Enumerating the 14,928 prediagrams of k = 2, ℓ = 3 takes
  1.4 s instead of 9.0 s (k = 3, ℓ = 2: 0.4 to 0.5 s instead of 1.2 to
  1.3 s). With `DAEDALUS_PREDIAGRAM_EAGER=0`, `ou_quartic` at k = 2 up to
  ℓ = 3 on 7 τ points takes 6.6 to 6.9 s instead of 14.4 to 15.0 s. The
  first call at a small cell is slower than before: for k ≤ 3 at ℓ ≤ 1,
  k ≤ 5 at ℓ = 0 and k = 1 at ℓ = 2 it takes about 5 to 90 ms instead of
  0.1 to 20 ms, because the compiled enumerator starts one tree-generator
  process per tree order, and later calls for the same cell take 0.1 to
  10 ms. From k = 4, ℓ = 1 and k = 6, ℓ = 0 up, the first call is already
  faster than the eager enumerator (k = 5, ℓ = 1: 0.8 s instead of 1.5 s),
  and rebuilding the records dominates later calls. End-to-end runs do not
  show the difference. Without the compiled extension (`DAEDALUS_FASTENUM=0`), a first call is
  1.1 to 1.5 times slower than the eager enumerator at ℓ ≤ 2, because it
  rebuilds the records from certificates (k = 4, ℓ = 1: 0.31 s instead of
  0.26 s). It is about as fast at k = 3, ℓ = 2 and at k = 1, ℓ = 3.
- Same diagrams, other representatives. The diagram set is unchanged, but
  each isomorphism class is represented by another labelled diagram, with
  other vertex numbers, in another order: sorted by certificate bytes, which
  depend on the certificate backend.
- Diagram order and vertex numbers of a temporal result. As before this
  change, the order of `result['diagrams']`, and the vertex numbers in each
  record's `typed_diagram` and in the keys of its `classify` entry
  (`vertex_time_factors`, `source_time_info`), depend on where the
  prediagrams came from. A cache-on and a cache-off run differ, and
  `DAEDALUS_PREDIAGRAM_EAGER=0` changes them again. `dd.export_tikz(result,
  index=i)` and the panel order of its array follow `result['diagrams']`, so
  the same index can draw another diagram in another run.
- `dd.export_tikz(..., symbolic_factors=True)` numbers the interaction
  factors v₁, v₂, … and the superscripts of same-order noise sources in
  sorted order of their LaTeX expressions. Before, it numbered them in order
  of first appearance in the diagram list, so the same model printed v₁ for
  different factors in a cache-on and a cache-off run: on
  `ou_quartic_two_dim_color_corr` (k = 2, ℓ ≤ 1), 3ε₁x*₁ was v₁ cache off
  and v₃ cache on. A figure exported earlier can carry other numbers.
- Spatial results with the default `grid` integrator move by rounding only.
  Measured on τ ∈ {0, 0.25, 0.75, 1.5} and χ ∈ {0.3, 0.9, 1.7, 3.0}, the
  eager and the streamed records differ by at most 2.4e−16 relative for
  `allen_cahn_1d_subcritical_infinite` (ℓ ≤ 2) and 2.6e−16 for
  `reaction_diffusion_2d` (ℓ ≤ 1), and not at all for
  `coupled_rd_2species_1d` (ℓ ≤ 1).
- Spatial sampling integrators (`SPATIAL_INTEGRATOR=mc` or `bessel`). Each
  diagram's Monte-Carlo seed now comes from its isomorphism class instead of
  its position in the diagram list (it was `1234 + index`). Reordering the
  list therefore no longer changes the estimate. Before (measured on the
  code before this change, which used the eager records), two shuffles of
  the list moved `allen_cahn_1d_subcritical_infinite` (ℓ ≤ 2, N = 2e4,
  τ ∈ {0, 0.5}, χ ∈ {0.5, 2}) by up to 1.5e−3 relative with `mc` and
  9.6e−4 with `bessel`: Monte-Carlo noise. The new seeds and the new
  representatives still move these estimates.
  - With `mc`, the move is Monte-Carlo noise. At ℓ ≤ 2 with N = 1e6,
    C(x, τ) moves by 1.4e−3 relative between the eager and the streamed
    records, inside the 4e−4 to 3.9e−3 spread between seeds.
  - With `bessel`, see "Fixed: `SPATIAL_INTEGRATOR=bessel` warns at τ ≠ 0"
    below: at τ = 0 the move is Monte-Carlo noise (both record sets are
    within 2.5e−3 of the `grid` value); at τ ≠ 0 the estimate is biased,
    and the bias depends on the representative. At τ = 0.5, the ℓ = 2 term
    is 25 % to 38 % below the `grid` value with the streamed records and
    5 % to 10 % below with the eager records, at N = 1e6 and 4e6 alike.
- Why temporal runs keep the eager records. On models whose integration
  regions fall back to `scipy.nquad`, another representative can send other
  regions there, so results would move by the fallback's error, far above
  rounding. Measured with `DAEDALUS_PREDIAGRAM_EAGER=0` against the default,
  with the `scipy.nquad` fallback at its default tolerance as it stands at
  this change (commit 16b5564). The call counts and the moves below are
  that fallback's, and any change to the fallback changes them:
  - `single_population_spike_reset_test` (parameters of the k = 2 table in
    "Moved (measured)"), k = 2, ℓ = 1, makes 20 fallback calls instead of 40.
    Its one-loop term moves by up to 1.0e−10 absolute (2.1e−6 relative, at
    τ = 10), and `C_tau` by up to 3.6e−7 relative. At every τ of that table
    the streamed value is within 1.4e−12 of the value computed with a tight
    fallback tolerance; the eager value is within 1.0e−10. With the tight
    tolerance, the two representatives agree to 1.7e−15 absolute.
  - `single_population_quad_exp_test` (`P_SP`), k = 2, ℓ = 1, makes 36
    fallback calls instead of 48. Its one-loop term moves by up to 1.1e−9
    absolute (5.3e−6 relative), and `C_tau` by up to 2.9e−7 relative. With
    a tight fallback tolerance, the two representatives agree to 4.4e−14
    absolute.
  - Where every region is analytic the move is rounding only: at most
    4.6e−16 relative for `ou_quartic` (k = 2, ℓ ≤ 3), and none for
    `linear_hawkes`, `multipopulation_test` and
    `single_population_linear_delta_spikes_test` (k = 2, ℓ ≤ 1, parameters
    `P_LINH`, `P_MP`, `P_LD` of `tests/tools/phase_j_zoo_baseline.py`;
    their one-loop terms are 0 there).

  `TEMPORAL_CACHE_OFF_STREAMS` in `engine/enumeration/prediagram_cache.py`
  switches the temporal default once the fallback agrees across
  representatives on every model that reaches it. The slow test
  `test_temporal_streamed_totals_match_eager_on_a_fallback_model` is the
  gate. With the fallback of commit d68e383 (measured 2026-10-06), the two
  `single_population_spike_reset_test` cases (per-diagram and grouped) pass:
  their totals agree to 1e−13 relative. `single_population_quad_exp_test`
  still differs, by 1.21e−11 relative, and stays a strict expected failure.
  The flag flips only when every case passes, so it stays False for now.
  Run the gate (`sage -python -m pytest
  -m slow tests/test_prediagram_cache.py -k fallback_model`, about 7 min)
  after any change to the Phase J fallback.
- Cache-on and cache-off runs can use different representatives, as before.
  The shipped prediagram files hold Sage-backend certificates, written on a
  Sage install with the optional bliss package. A temporal cache-off run
  uses the eager records. With the fallback of commit 16b5564 (as above),
  they differ from a fresh clone's cache-on records by 4.5e−15 relative in
  `C_tau` for `single_population_spike_reset_test` and 8.0e−11 for `single_population_quad_exp_test` (k = 2, ℓ = 1; the
  shipped and the eager representatives take different quadrature routes
  there, 24 and 48 fallback calls). With `DAEDALUS_PREDIAGRAM_EAGER=0` the
  cache-off records are the streamed ones: they differ from the shipped
  ones by 3.6e−7 and 2.9e−7 where the compiled extension builds, and are
  identical to them with the Sage backend (`DAEDALUS_FASTENUM=0` or
  `DAEDALUS_CERT_BACKEND=sage`) on a Sage install with bliss.

### Performance: memo for the chain integral with intermediate uppers (numbers unchanged)

`_chain_with_intermediate_uppers`, the chain-simplex integral of the m ≥ 3
poset path, is a pure function of its arguments and the Phase J poset walk
calls it many times with identical ones (the m ≥ 3 chain path at higher loop
order; models with repeated poles). It is now memoised. A hit returns the
object the first call computed, so every result is bit-identical to the memo
off (`np.array_equal`, checked on the models below).

- Flag `USE_CHAIN_UPPERS_MEMO` (default on, read at call time); environment
  `DAEDALUS_CHAIN_UPPERS_MEMO=1|0`, any other value is an error. It is a pure
  speed switch: `DAEDALUS_PHASE_J_LEGACY` does not touch it.
- The key is the chain's alphas (exact complex values), `L`, the uppers
  (sorted, so dict order does not matter), the chain-top upper and the module
  state the result depends on, read at call time: `USE_NUMBA_CHAIN_SIMPLEX`,
  `USE_CHAIN_SIMPLEX_PRECISION_FIX` (and its threshold),
  `USE_POSET_MPMATH_ACCUMULATION`, `USE_POSET_CAP_MATCH_SCIPY`. Toggling any
  of them misses the memo.
- The existing per-simplex tables (`_chain_simplex_memo_fast/_poly`) keyed on
  the arguments only, so a toggle of `USE_NUMBA_CHAIN_SIMPLEX` or
  `USE_CHAIN_SIMPLEX_PRECISION_FIX` could be answered from a stale entry. The
  fast table now keys on that state. `_chain_simplex_memo_clear()` also clears
  the new table.
- Thread safe (the spatial path runs Phase J in a thread pool): lookup and
  insert hold a lock, the value is computed outside it. At 200,000 entries the
  table is cleared in one locked step, so memory stays bounded.
- Counters `chain_uppers_memo_hits`, `chain_uppers_memo_misses` and
  `chain_uppers_memo_evictions` in `_RUNTIME_COUNTERS`. With the memo on the
  inner counters `chain_simplex_memo_hits` and `chain_simplex_*_returned_none`
  count fewer calls, because most calls never reach the inner tables.
- The poset plan loop builds the (L, uppers, chain-top) key part once per
  linear extension instead of once per pole-tuple group.

Measured on a 4-vCPU Linux VM with other jobs running on it, memo off → on, whole
run including diagram setup (the memo does not touch the setup, which is most
of what remains; the two OU runs give two samples each):

| run | off | on | hits / misses |
|---|---|---|---|
| `ou_quartic_two_dim`, k = 2, ℓ = 2, 7 τ | 46.3–49.4 s | 28.1–29.2 s | 2,613,114 / 54,726 |
| `ou_quartic`, k = 4, ℓ = 2, 1 point | 36.8–38.4 s | 19.4–20.1 s | 473,592 / 40,248 |
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | 89.2 s | 42.0 s | 7,988,455 / 584,321 (cap reached) |
| `single_population_spike_reset_test`, k = 2, ℓ = 1 | 14.7 s | 12.3 s | 173,278 / 3,922 |
| same, grouped | 6.3 s | 4.6 s | 2,958 / 3,922 |

### Performance: per-diagram setup levers (numbers unchanged)

After the chain memo, most of the time of a run with many diagrams went into the
per-diagram setup of `integrate_diagram`: on `ou_quartic_two_dim` (k = 2, ℓ = 2)
5217 of the 5347 diagrams have a prefactor that is numerically zero and still paid
the full setup, and the model-level data (`build_G_t_matrix`, the pole and residue
arrays) was rebuilt for every diagram. Four independent levers cut that cost. None
moves a number: with a lever on, every total, per-ℓ value and per-diagram value is
bit-identical (`np.array_equal`) to the lever off, in one process (checked on the
models below and in `tests/test_phase_j_setup_fastpath.py`).

- **L1 zero-prefactor exit** (`final_integral.USE_SETUP_ZERO_EXIT`,
  `DAEDALUS_SETUP_ZERO_EXIT`, counter `setup_zero_exit`). A diagram with no
  noise-source (`cumulant_specs`) vertex and no `ConvVertexType` vertex whose numeric
  prefactor is a plain number equal to exactly 0 returns a zero contribution before
  the G(t) matrix and the per-edge setup are built. It stays in the list at its
  position (a zero entry), and its result has the keys and types of the full path;
  `stripped_integrand`, `constraints` and `edge_info` are filled on first read by
  running the full path. It declines (full path as before) when `external_fields` is
  missing or does not match the leaves, when the propagator has no pole or incomplete
  residue data, when the prefactor does not evaluate to a number, or while the
  `_SUBSET_HOOK` debugging hook is set. Two observational differences remain for a
  diagram with a zero prefactor: the full path also emits zero-coefficient
  `delta_contributions` entries (and `shotnoise` / `forced_delta_pruned` diagnostics)
  for its shot-noise subsets, which the early exit does not; no value computed from
  them changes, and nothing in `api/` or `engine/` reads them.
- **L2 model-level data once per call** (`USE_SETUP_PROP_TD`,
  `DAEDALUS_SETUP_PROP_TD`, counters `setup_prop_td_used`, `_stale`, `_g_t_builds`,
  `_entry_builds`). `compute_correction_td` builds one `PropagatorTD` (the G(t)
  matrix, the poles, the `(residue, λ)` tuples per propagator entry and the delta
  coefficients, each lazily and with the same conversions as before) and passes it
  to `integrate_diagram(prop_td=...)`, a new optional argument; `None` builds
  everything per diagram as before. The object lives for one call and is recognised
  by the identity of the propagator data, `num_params` and the pole / residue lists
  it was built from; a mismatch builds locally. There is no `id()`-keyed global, so
  the spatial bridge, which re-solves the poles for each q, cannot be served stale
  data. Do not mutate those objects in place while a call runs.
- **L3 lazy SR** (`USE_SETUP_LAZY_SR`, `DAEDALUS_SETUP_LAZY_SR`, counters
  `setup_lazy_sr_deferred` / `_built`, `setup_lazy_display_deferred` / `_built`).
  `edge_info[i]['smooth_factor']` is read only by the shot-noise and SR-integrand
  branches, and the display-only `stripped_integrand` by nothing; both are built on
  first read. The keys stay present in every record (the records are a dict
  subclass whose listed keys resolve on read); a failed result carries the eager
  value.
- **L5 shared automorphism work** (`engine.diagrams.symmetry.USE_AUT_MEMO`,
  `DAEDALUS_SETUP_AUT_MEMO`, counters `setup_aut_memo_hits` / `_misses` /
  `_evictions`, `setup_aut_wick_skipped`). `_automorphism_order` is memoised per
  (diagram, `fix_external`): the entry holds the diagram (its `id` cannot be reused),
  a hit checks identity and the sizes of its containers, the table is cleared at
  8192 entries and is thread safe. `external_wick_compensation` is skipped when the
  diagram has a single Wick mapping, where the index |Aut_free| / |Aut_fixed| is
  exactly 1 (every external field occurs once, so the leaves are distinguished by
  field with or without fixing them; checked against the real group computation on
  10,892 diagrams). Do not mutate a typed diagram after it was classified.
- Each flag is a module boolean read at call time; the environment variable is read
  at import (1 or 0, also true/false, yes/no, on/off; anything else is an error).
  `DAEDALUS_PHASE_J_LEGACY_SETUP=1`, read at call time, forces all four off. It is a
  different switch from `DAEDALUS_PHASE_J_LEGACY`, which reproduces older numbers;
  neither touches the other, and the setup levers do not change any number.

Measured on a 4-vCPU Linux VM (other jobs running; off and on alternated in one
process, warm disk cache, whole `compute_cumulants` run including enumeration and the
τ evaluation; each pair is two samples):

| run | all off | all on |
|---|---|---|
| `ou_quartic_two_dim`, k = 2, ℓ = 2, 7 τ | 35.0 / 35.2 s | 10.8 / 11.2 s |
| `ou_quartic`, k = 2, ℓ = 3, 7 τ | 6.7 / 6.9 s | 1.9 / 1.9 s |
| `ou_quartic`, k = 4, ℓ = 2, 2 points | 29.4 / 29.5 s | 16.2 / 16.4 s |
| `ou_quartic`, k = 4, ℓ = 1 | 0.61 / 0.60 s | 0.36 / 0.39 s |
| `single_population_spike_reset_test`, k = 2, ℓ = 1 | 14.0 / 13.7 s | 10.2 / 10.6 s |

Under `cProfile` the cumulative time of `integrate_diagram` for the first row (the
per-diagram setup; the evaluation happens later) drops from 31.1 s to 1.6 s of a
45.8 s → 16.1 s run. L1 carries almost all of it on this model; with L1 on, L2, L3
and L5 each save about a second or less there, and more on models where few diagrams
have a zero prefactor. Models with no zero-prefactor diagram (the spike models) gain
only a few percent.

### Performance: δ-edge elimination without Maxima (numbers unchanged)

For each δ-subset, `integrate_diagram` sets `dt_e = 0` for every δ edge and solves it
for an integration variable. It did this with Sage `solve` (Maxima), and it tested
variables and residual equalities with SR `==` and `is_zero()`, whose `bool` can run
randomized zero proofs (`random_element`). On every model in the repo these equations
are linear forms in the vertex times with exactly rational coefficients. The new
lever solves them by exact linear algebra in QQ: for `a·x + rest = 0` it returns
`x = -rest/a`, rebuilt as an SR sum of `Rational · symbol` terms. Pynac stores that
sum in the same canonical form as the `solve` result, so the substitution is the same
polynomial and the same SR tree. Every later `.subs`, `.coefficient` and `float` then
sees the same expression.

- **The lever** (`final_integral.USE_SETUP_DELTA_SOLVE`, `DAEDALUS_SETUP_DELTA_SOLVE`,
  default on; counters `setup_delta_solve_fast` / `setup_delta_solve_fallback`).
  Variables are matched by name and never with SR `==`. Anything that is not an
  exactly rational linear form in the eliminated variable goes to `sage_solve` exactly
  as before. That covers a float, complex or symbolic coefficient, a nonlinear term, a
  constant such as π, a zero coefficient, and an exception. The chain resolution of
  earlier substitutions is unchanged. The residual external-time zero test is now
  exact on the coefficients, and keeps the SR test for anything else. No floating
  point enters the elimination, so a coefficient or pair shift that is exactly 0 comes
  out as exactly `0.0`, never a 1e-17 residue. The `_const_row_verdict` / Θ(0)
  decisions depend on that. It follows the M5 lever conventions: module boolean read
  at call time, environment variable read at import, and
  `DAEDALUS_PHASE_J_LEGACY_SETUP=1` forces it off.
- **Validation mode** (`final_integral.VALIDATE_DELTA`, or
  `DAEDALUS_PHASE_J_VALIDATE_DELTA=1` at call time; default off; counter
  `setup_delta_validated`). It also solves every subset the legacy way and requires
  exact agreement: the same eliminated and remaining variables, and every substitution
  equal as a polynomial and as an SR tree. It also requires the same residual
  equalities and verdicts, and `==`-equal constraint rows (`a_int`, `a_ext`, `c0`)
  for every smooth edge. Any difference raises `DeltaSolveValidationError`, naming the
  diagram serial, the subset and its δ edges.
- **Grouped path.** `integrate_grouped_diagram` has its own δ-solve
  (`grouped_integral._grouped_delta_solve`); it now reads the same lever and umbrella
  and calls the same `_delta_eliminate_exact` (same elimination order, by-name variable
  matching, `sage_solve` fallback, counters). Like the legacy grouped solve it does not
  chain-resolve earlier substitutions. The residual shot-noise test
  (`_has_nontrivial_equality`) uses the exact zero test with the SR `is_zero()` as the
  fallback. The same validation mode compares the grouped solve to the legacy one
  (feasibility, substitutions as polynomials and SR trees, residual equalities and
  their zero verdicts) and raises `DeltaSolveValidationError` naming the subset. The
  M2a token replay `_delta_solve_leaves_residual` is unchanged and still agrees with
  the solve (`tests/test_grouped_delta_solve.py`). On the grouped
  `single_population_spike_reset_test` (k = 2, ℓ = 1) the lever removes the roughly
  1.5 s of `sage_solve` from a 6.3 s run. Totals, per-ℓ values and per-group values
  are `np.array_equal` lever off / on / validate.

Checked on a 4-vCPU Linux VM with the validation mode and with the lever off vs on in
one process (warm disk cache). Every total, per-ℓ value and per-diagram value was
`np.array_equal`:

| run | eliminations validated (fast / fallback) | mismatches |
|---|---|---|
| `single_population_spike_reset_test`, k = 2, ℓ = 1 | 622 (622 / 0) | 0 |
| `single_population_spike_reset_test`, k = 1, ℓ = 2 | 942 (942 / 0) | 0 |
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | 188 (188 / 0) | 0 |
| `linear_hawkes`, `multipopulation_test`, `single_population_linear_delta_spikes_test`, `single_population_spike_reset_test` k = 1 (k = 2 where not stated, ℓ = 1) | 2 each (2 / 0) | 0 |

`ou_quartic` (k = 2, ℓ = 2), `ou_quartic_colored` and `ou_quartic_two_dim_color_corr`
reach no δ-solve and are unchanged. Whole `compute_cumulants` run, lever off → on
(other jobs running on the VM):

| run | wall off → on | `sage_solve` share of `integrate_diagram` (off) | `random_element` calls off → on |
|---|---|---|---|
| `single_population_spike_reset_test`, k = 2, ℓ = 1 | 14.0 → 3.4 s | 82 % | 3392 → 160 |
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | 457 → 444 s (heavy contention; `integrate_diagram` is 24 s of it) | 12 % | 1488 → 480 |
| `multipopulation_test`, k = 2, ℓ = 1 | 3.7 → 4.2 s (noise) | 33 % | 1600 → 1600 |

Under `cProfile`, the cumulative time of `integrate_diagram` on the first row drops
from 9.0 s to 1.4 s. On the other two rows the δ-solve was already a small part of
the run. The remaining `random_element` calls come from outside the δ-solve.

### Fixed: `SPATIAL_INTEGRATOR=bessel` warns at τ ≠ 0

The `bessel` integrator is exact only when the external times are equal
(k = 2: τ = 0). Its radial Bessel-K reduction treats every edge weight as
proportional to the radial variable, which fails for the edges to the
external legs when the external times differ, and it never places an
internal vertex later than the earliest external time. At τ ≠ 0 its loop
terms are therefore biased, by an amount that depends on which leaf the
diagram's labelling puts at τ. Measured on the one-loop diagram of
`allen_cahn_1d_subcritical_infinite` at x = 1 (N = 1e6): within 3e−4 of the
`grid` value at τ = 0; 17 % to 78 % below it at τ = 0.25 to 1, where `mc`
agrees with `grid` to 1.4 % or better. On the full model (ℓ ≤ 2) at
τ = 0.5, C(x, τ) is 4.5 % to 8.8 % below the `grid` value, depending on the
prediagram source. This predates 0.2.0.

The integrator now issues a `BesselUnequalTimesWarning` (a `RuntimeWarning`)
for a loop diagram at unequal external times; the value is unchanged. Use
`SPATIAL_INTEGRATOR=grid` (the default) or `mc` at τ ≠ 0. The strict
expected failure `test_bessel_matches_grid_at_unequal_external_times` keeps
the bias measured.

### Fixed: `dd.generate_report`

- The cover page raised `KeyError: 'nstar'` for every model without `n`/`v`
  saddles, for example `ou_quartic`. It now lists the `n`, `v` and `m`
  saddles a model has, then every other saddle it declares.
- Every per-diagram page showed "(plot error: 'phase_j_result')" instead of
  a curve: the panel read a result key that `compute_cumulants` never
  returns. Each k = 2 page now plots the diagram's own contribution, matched
  to the diagram by identity, never by page number. Like `C_tau`, it is
  sampled with the Itô left limit at τ = 0. The grouped Phase J path has no
  per-diagram contributions, so its pages show none.
- Diagram pages come in a fixed order: by loop order, then by isomorphism
  class. They number the vertices canonically, so "Diagram i / N", the
  vertex list and the drawing depend only on the diagram's class, not on the
  prediagram source or the list order. Measured on `ou_quartic` (k = 2,
  ℓ ≤ 2, 71 diagram pages): eager, streamed and shuffled records give the
  same text on every page and the same raster image at 40 dpi; the curves
  agree to 5.9e−16.
- When several edges join the same two vertices, the edge label lists their
  distinct propagator pairs. Before, it showed whichever edge was drawn last.

### Changed: Phase J skips identically-zero work (numbers unchanged)

Phase J no longer builds or integrates two kinds of work whose result is
exactly 0, and it checks a propagator that has no pole at all before using
it. All three are controlled by the new call-time flag `STRUCTURAL_ZEROS`
(on by default; see Added).

- **Purely instantaneous propagator entries.** A propagator entry with a δ
  part but no smooth part, for example the spike train of a population that
  receives no input, makes every δ-subset that keeps its edge smooth
  identically zero. Such a subset is no longer built, per diagram or in
  grouped builds. In a grouped build a subset is skipped only when every
  contributing typed diagram keeps such an edge smooth. An entry counts as
  purely instantaneous only when this is proven: every stored pole residue
  of the entry is exactly 0, and the entry of the frequency-domain
  propagator `G_ft`, with the numeric parameters substituted, does not
  contain ω. The second test looks at the propagator before the builder
  drops any pole (it keeps only poles with Im ω > 1e−9), so an entry whose
  residues are 0 only because its own mode is marginal, or because the
  builder set a tiny residue to 0, is not skipped. Where `G_ft` is not
  available, the stored residues decide. The builder does not compute
  `G_ft` for a propagator matrix of size 6 or more, or with more than 20
  free symbols (for example `multipopulation_test`), nor when its symbolic
  inverse exceeds its time budget or fails. Every evaluator reads those
  same residues, so the skipped work is exactly the 0 it would have
  computed. (For example, `multipopulation_test` with the second row of
  each of its four coupling matrices set to 0, k = 2 up to one loop, skips
  5 δ-subsets in all this way on the per-diagram path, and 2 in a grouped
  build.) Only an exact 0 counts: a tiny nonzero residue is never treated
  as 0, and neither is a NaN or infinite one, so a corrupt
  propagator entry still shows as NaN in the result instead of being
  skipped. Example: `single_population_linear_delta_spikes_test` with
  `w = [[0, 0.25], [0, 0]]` (population 2 receives no input), tree level.
  In the per-diagram path, 31 of the 33 δ-subsets of the 10 typed k = 3
  diagrams are skipped, and at k = 4, 344 of the 346 δ-subsets of the 68
  typed diagrams. A grouped k = 4 build skips 105 subsets in its 12 groups
  (of the 140 that have a contributing typed diagram), and 256 of the 258
  δ-subsets of the 48 typed diagrams it evaluates per diagram; its counter
  `forced_delta_pruned` adds both (361). A k = 4 run (build plus six
  evaluations) takes 0.7 to 0.9 s instead of 2.5 to 2.7 s on the
  per-diagram path (0.9 to 1.1 s instead of 2.5 to 3.0 s grouped; measured
  on a loaded machine).
  At the parameters of the tables above no public model has such an entry,
  so their runs are unaffected.
- **Empty order cycles at the `scipy.nquad` fallback.** The fallback returns
  0 at once for a region whose ordering rows contain a directed cycle with
  shifts summing to ≤ 0, in exact arithmetic. It used to integrate such a
  region, and with fast poles could raise `OverflowError` on it. With the
  default settings the exact emptiness tests of the Itô equal-time section
  above answer these regions first; the fallback test is a safety net for
  `THETA0_CONST_ROW_MODE = 'legacy_clip'`, for the two-time analytic
  integrator switched off, and for the symbolic path.
- **Propagators without any pole.** Because the builder keeps only strictly
  retarded poles, a marginal or non-retarded mode, for example λ = 0 at
  q = 0 for a massless field, is dropped, and a propagator can reach Phase J
  with an empty pole list although it has a smooth part. Such a propagator
  is now decided from `G_ft` alone, with the numeric parameters
  substituted:
  - An entry that does not contain ω is purely instantaneous, and is
    handled as above. For example, `single_population_linear_delta_spikes_test`
    with every coupling 0 (two independent constant-rate spike trains) now
    gives exactly 0 at every point of its k = 2 τ grid and at k = 3, per
    diagram and grouped. Before, these runs raised a bare `IndexError`,
    except ⟨n₁ n₂⟩ at k = 2 in a grouped build, which already gave 0.
  - If a diagram uses any other entry, or `G_ft` is not available, Phase J
    raises `PoleFreePropagatorError` (a `ValueError` subclass, in
    `engine.integration.time_domain.final_integral`), naming the entry and
    the reason, instead of returning a value. Before, the per-diagram path
    raised a bare `IndexError` there, and the grouped path could return a
    value in which that smooth part was silently 0. The two massless
    spatial probes of the model zoo (`edwards_wilkinson_1d` and
    `reaction_diffusion_conserved_1d` at μ = 0) now stop with
    `PoleFreePropagatorError` at their q = 0 sample instead of
    `IndexError`.

Values are bit-identical to the code before this change, compared in one
process on 56 public configurations (per diagram and grouped counted
separately; `ou_quartic`, `ou_quartic_colored`, `linear_hawkes`,
`single_population_spike_reset_test` also with population 2 receiving no
input, `single_population_linear_delta_spikes_test` also with the one-way
and the uncoupled parameters above, `single_population_quad_exp_test` also
with population 2 receiving no input, `multipopulation_test` also with
the second rows, or all, of its coupling matrices set to 0,
`dendritic_quad_soma_sigmoid`, and seven spatial entries of the model zoo),
with the flag on, with it off, and under `DAEDALUS_PHASE_J_LEGACY=1`.
The only differences are where the old code raised: the uncoupled linear
delta spikes (`IndexError`; now exact zeros) and the massless spatial probes
(`IndexError`; now `PoleFreePropagatorError`). Hand-built cases add the
`OverflowError` of the fallback and the grouped path's silently dropped
smooth part. With `STRUCTURAL_ZEROS = False` the old behaviour returns,
bit-for-bit, including those errors.

Developer-facing changes (values do not change):

- `subset_diagnostics` can contain entries with status
  `'forced_delta_pruned'`;
- `n_subsets_evaluated`, and the `subset_index` of `_SUBSET_HOOK` payloads,
  count only the subsets that are built;
- `delta_contributions` no longer lists the zero-coefficient entries of
  skipped shot-noise subsets;
- a region answered by the cycle test still counts in `nquad_calls`, like
  the Θ(0) answers given at the entry of the fallback;
- the model-zoo baseline (`tests/fixtures/phase_j_legacy_baseline.*`,
  recorded before this change) lists the two massless spatial probes as
  `IndexError`; a re-run now reports `PoleFreePropagatorError` for them.

### Changed: hardened quadrature fallback (numbers move at the quadrature tolerance)

When no analytic Phase J integrator answers an integration region (for
example a two-time region whose analytic formula would overflow), Phase J
integrates it with scipy quadrature. Before this change that fallback was
`scipy.nquad` with scipy's default tolerances (absolute 1.49e−8), no
breakpoints, and a ±200 box for every integration time whose bounds depend
on other integration times. A narrow peak could fall between all the nodes
of the first quadrature panel: both quadrature rules then return about 0,
the error estimate is about 0, and the whole region is lost. On wider peaks
the fallback was accurate only to about 1e−5 relative. An empty inner
interval was still sampled, at points outside the region, where fast decay
rates overflow.

The fallback now:

- integrates every time over the region's own bounds, widened by one unit
  in the last place: the exact projection of the region onto that time
  given the outer ones. For ordering rows (rows of the form
  ±(s_i − s_j) + c > 0 or ±s_i + c > 0) this is their closure by shortest
  paths in exact rational arithmetic. If any other row is present, every
  row enters an exact Fourier–Motzkin elimination (rational arithmetic),
  which gives the projection whatever the rows; above 4096 rows the
  elimination stops (counted) and such rows bound only the time where they
  involve no inner time. Every row is also checked pointwise. A region
  whose rows cannot all hold is 0 without quadrature;
- returns 0 for an empty interval without evaluating the integrand, and
  never evaluates the integrand where a row is ≤ 0 (Θ(0) = 0);
- integrates the innermost time in closed form when the integrand's
  exponential modes are known, so the quadrature runs over one time fewer.
  The per-diagram and grouped evaluators supply the modes; regions without
  them (the evaluator for non-rational noise kernels, a grouped subset
  without pole data, direct callers) are counted and treated as described
  below. The integrand's expansion in that time (per diagram, one term per
  combination of modes of the edges that contain it; grouped, one per pole
  tuple) is built once per region and evaluated with numpy when it has
  many terms. Above 65536 such per-diagram terms that time is integrated
  by quadrature instead (counted);
- truncates a direction that the rows leave open at K/κ (K = 40) beyond
  the farthest point on that side where the integrand can change shape:
  the interval's finite end, a kink of the inner integral or an external
  time. κ is a decay rate certified along that direction. Every mode must
  decay (Re λ < 0); then the integrand is bounded by S·e^{−Σ κ_e Δt_e},
  with κ_e the slowest rate of edge e and S a constant.
  - When every row is an ordering row with coefficients ±1 and is an
    edge, and every edge is a row (every Phase J region measured), κ is
    the slowest decay rate of the modes. In that case the tail beyond the
    outermost cut, at the distance d from the finite end, is at most
    S·κ^{−m}·e^{−K}·Σ_{j<m} K^j/j! with K = κd, for m integration times
    and whatever the shape of the inner slices. At the first cut that is
    1.7e−16·S/κ^m at m = 2, 3.6e−15 at m = 3 and 2.8e−11 at m = 7. (Use
    the values of a spanning tree of rows as coordinates. Its path from
    that time to its finite end adds up to d, and every tree value decays
    at least at the rate κ.)
  - With any other rows (other coefficients, rows that are not edges),
    linear programs give the rate at which the minimum of Σ κ_e Δt_e over
    a slice grows along each open side of each time. κ is the smallest of
    these rates, and it is no larger than the slowest mode's rate times
    the smallest coefficient. The rate therefore depends on how the rows
    combine, not on how a row is scaled. Before this, a region whose rows
    allow e^{s/4} along s, written with an integer row (s − 4s' > 0), was
    cut at 40 instead of 160 and came out 5e−5 low, with no flag; with
    (s − 8s' > 0) it was 7e−3 low. The outermost tail is then bounded
    from the slices' widths, which are measured beyond the cut. A slice
    that is unbounded in an inner time is bounded through the region's
    box fibration. Before this, the bound assumed chain-shaped slices:
    1024-times-wider slices of a cancelling integrand left it 9e−10 low.
    If an edge can be negative on the region, a side does not decay,
    or the slices cannot be bounded, the region has no certificate.

  These bounds are relative to S, not to the integral, which can be far
  smaller: an integral whose mass sits 60 time units from the finite end
  is about e^{−40}·S. So after integrating, the outermost cut's tail bound
  is compared with 1e−11 times the result. If it is larger, the cut moves
  out, and the strip between the two cuts is integrated and added
  (counted).
  - With rows of the second kind the inner times' cuts are checked as
    well. Every point that a cut drops, at any time, lies K/κ beyond the
    farthest vertex of its slice, so Σ κ_e Δt_e there exceeds its minimum
    over the region by at least K. Linear programs bound the integral of
    S·e^{−Σ κ_e Δt_e} over all such points: the region's widths where
    that sum stays below a level, which grow at most linearly with the
    level. If the bound exceeds 1e−11 times the result, the inner times'
    K moves out (the outermost cut at least as far) and the region is
    integrated again (counted). Before, the inner cuts were not checked.
    With slices that widen with the distance (rows that are not edges)
    and modes that cancel, the dropped part could exceed the tolerance by
    far: a model-free region with m = 4 integration times whose modes
    cancel to 2^−21 of their parts was 1.5e−8 low, and to 2^−25, 2.5e−7
    low (exact values from Mathematica). Those regions are now
    1.3e−10 and 4.4e−9 off; the rest is the rounding of their cancelling
    inner values, which QUADPACK's roundoff warning reports (counted and
    warned about). The cuts stay unchecked where an inner time's
    vertices were not all enumerated or linear programming gives no
    bound; that is counted (`nquad_hardened_inner_unchecked`) but not
    warned about, so such a region can be off by the amounts above with
    at most QUADPACK's roundoff warning, which comes from the rounding,
    not from the cut.
  - With ±1 ordering rows that are edges, as in every Phase J region
    measured, the same check runs, with the linear-programming bound
    built only when an inner time was actually cut. Without it, a
    model-free region of this kind (an open inner time holding a chain of
    edges, inner widths 2^−22 to 2^−24) lost up to 6e−8 of its value at
    K = 40 with nothing counted; with it, those regions are within 1e−8.
    On the public models measured, Phase J values are unchanged by this
    check.

  If some mode does not decay, the modes are not known, or the
  certificate fails, the old distance 200 is used, counted and reported
  in the warning;
- uses a relative tolerance of 1e−10 and an absolute tolerance of 1e−13
  times a scale. The first scale is the smaller of a bound of the
  integrand derived from the modes (each edge bounded separately over a
  box around the region) and the largest value of the integrand at 64
  points spread over the region (computed from the modes when they are
  known), but with known modes never below 1e−3 times the bound. Both can
  exceed the integral by many orders of magnitude: close poles (residues
  ±1/ε that cancel), oscillation, decay across the region. QUADPACK then
  stops at its first estimates. So if the absolute tolerance used is more
  than 10 times 1e−13 times the result, the region is integrated again
  with 1e−13 times the result (floored at 1e−22 times the sampled largest
  value, so that regions whose value lies far below rounding, such as
  near-coincident time points, are not chased; counted). For a thin
  region that floor can allow more than 1e−8 relative error; such a
  region is then counted (`nquad_hardened_floor_limited`) and named in the
  aggregated warning, so it is never accepted silently. Resolving thin
  regions exactly is left to the planned box-free integration. The sample alone can also sit many orders below the
  integrand when the integrand peaks at a finite end or a kink and the
  points are spread over a long truncated side. A model-free region with
  m = 3 times had a sampled largest value of 1.7e−12, where the integrand
  is about 0.4 and its bound is 4. That gave an absolute tolerance of
  1.7e−25, below the rounding noise of the imaginary parts of the level
  values computed from pole tuples. Its grouped evaluation did not finish
  in 25 minutes (per diagram: 0.3 s); it now takes 0.3 s and is exact to
  1.2e−15. The floor and the ordered real and imaginary passes of the
  next item each fix it on their own: with both, floors of 1e−3, 1e−6,
  1e−9 and 0 all take 0.3 s; with the floor alone, 1e−6 and below chased
  the noise again. A variant with ±1 ordering rows that are edges, as in
  Phase J, and the outer edge decaying at 1/16 ran past five million
  integrand evaluations grouped (35 s, then stopped); it now takes 0.4 s
  and is exact to 3.6e−15. In the public spike-reset
  k = 2 regions the sample sat at a median 4.5e−11 of the bound, while
  the integrals are about 1e−2 of it; they ran fast before. With the
  floor, the values of the captured public regions move by at most
  3.5e−15 relative, and some regions take a second pass (spike-reset
  k = 4: 288 of 1792 instead of 192);
- integrates the real and the imaginary part of each level separately,
  over one cache of complex values. The part that is larger at the first
  quadrature node goes first; the other is integrated with its absolute
  tolerance raised to 1e−10 times the first part's value; a call that ends
  with a QUADPACK warning is accepted when its error estimate is within
  1e−10 times |re + i·im| (or the absolute tolerance). A part that is only
  the rounding noise of the other, such as the imaginary part of a real
  integrand computed from pole tuples (or the real part of an imaginary
  one), then does not drive the subdivision. Judging the parts against
  the complex value was not enough on its own: with the floor removed,
  the grouped model-free region above, times 1 or times i, ran past three
  million integrand evaluations; it now takes 0.3 s and is exact to
  1.2e−15. A multi-peak oscillating region with general rows takes 5.8 s
  grouped and is exact to 1.2e−13 (before these two changes: past five
  million integrand evaluations in 240 s, then stopped). A real
  part does not depend on how the imaginary parts are integrated, so
  this ordering moves only the imaginary rounding noise of the public
  configurations (measured: real parts bit-for-bit unchanged);
- starts each quadrature call with a limit of 200 subintervals (at least
  twice its breakpoints). When the modes oscillate, the limit is 200 plus
  the number of half-periods over the interval, if that number exceeds
  100. A call that reaches its limit is repeated once with a limit of
  12800. A call that still ends with a QUADPACK warning is counted and
  reported by a `PhaseJNquadFallbackWarning` when its error estimate is
  above its own tolerance and also, multiplied by the widths of the
  enclosing levels' intervals, above 1e−10 times the region's scale (the
  result, in a second pass). An inner call of a second pass asks for
  1e−13 times the result in absolute terms, which the rounding of its own
  values can prevent with no loss at the region's tolerance. A call that
  stops at the limit 12800 is always counted;
- places breakpoints at the external times, at every kink of the inner
  integral, and geometrically spaced towards the ends of wide panels
  (with known modes). A narrow peak sits at, or close to, an interval end
  or a kink; a kink that is not a breakpoint can leave the whole peak
  inside a wide panel, unseen by every node. The kinks at a time are the
  coordinates of the vertices of the region's slice over the inner times.
  With ordering rows only they come from the closure (the alternating
  sums of its entries along paths through the inner times); with any
  other row the vertices are enumerated at each step (above 20000 row
  subsets at a step, that step keeps the closure's kinks, counted).

The old fallback is still available, bit-for-bit, with the flag
`NQUAD_HARDENED = False` (see Added).

**Moved (measured).** Values move only where the fallback is used, by the
error of the old fallback. Per region, the comparison is with the exact
integral of the same region (a 50-digit closed-form evaluation; tolerance
max(1e−8 relative, 1e−14 absolute)):

| model, configuration | regions served by the fallback | old fallback: outside the tolerance, largest error | new fallback: largest error |
|---|---|---|---|
| `single_population_spike_reset_test`, k = 2, ℓ = 1, τ = 0, 1, 3, 5, 10 | 40 per diagram (grouped: 10) | 18 of 40, up to 6.8e−5 relative, 1.5e−10 absolute | 1.1e−14 relative (grouped 1.9e−14) |
| same model and configuration, τ = 15, 25, 40, 60 | 32 (grouped: 8) | 8 of 32, some regions lost entirely (relative error 1.0, up to 4.0e−8 absolute; grouped 2 of 8) | 4.0e−15 relative (grouped 3.0e−15) |
| same model, k = 3 tree, ⟨n₁ n₂ n₁⟩ at the 9 points listed below | 360 (grouped: 162) | 70 of 360, up to 2.1e−6 relative, 1.4e−10 absolute (grouped 33 of 162) | 6.0e−15 relative (grouped 1.2e−15) |
| `single_population_quad_exp_test`, k = 2, ℓ = 1, τ = 0, 2.5, 10 | 48 (grouped: 12) | 34 of 48, up to 1.7e−4 relative, 2.7e−10 absolute (grouped 10 of 12, 1.8e−5) | 1.2e−15 relative (grouped 4.6e−16; reference boxes 400, 1500 and 3000 agree) |

(Parameters as in the tables above; `single_population_quad_exp_test`:
`Em = [0.8, 0.78]`, `tau = [10, 9]`, `a = [0.44, 0.44]`, the remaining
parameters at their defaults, as in `tests/tools/phase_j_zoo_baseline.py`.
The number of regions that reach the fallback depends on the diagram
cache, which selects the diagram representatives, and so does the old
fallback's error; the new values do not (measured with both caches: to
≤ 1.6e−14 relative for the `multipopulation_test` and spike-reset k = 3
and k = 4 values below). The counts and the old values in this section are from a
developer cache unless stated otherwise. A fresh cache sent 24 per-diagram
`single_population_quad_exp_test` regions instead of 48, and 216 (grouped:
72) spike-reset k = 3 tree regions instead of 360 (162).)

Effect on the results:

- `single_population_spike_reset_test`, k = 2, ℓ = 1, ⟨n₁ n₂⟩: the one-loop
  term moves by at most 1.0e−10 absolute. Relative to it that is
  ≤ 9.4e−9 at τ = 0, 1, 3 and 5, and 2.2e−6 at τ = 10 (+4.65705e−5 →
  +4.65706e−5); the total moves by ≤ 3.7e−7 relative (τ = 10:
  −2.720795e−4 → −2.720794e−4). The tree level is unchanged. Per-diagram
  and grouped Phase J now agree to 5e−15 relative (before: 1.4e−8 on the
  one-loop term). Farther out the old fallback's absolute tolerance
  (1.49e−8) was comparable to the values: at τ = 15, 25, 40 and 60 the
  one-loop term moves by 0.97 %, 1.9 %, 1.3 % and 0.84 % per diagram
  (τ = 60: +1.045104e−18 → +1.053871e−18; grouped the same except
  τ = 15, 1.6e−8), and the total, a cancellation of the tree and the
  one-loop terms there, by up to 16 % (τ = 60: −5.578e−20 → −4.702e−20).
- The same model, k = 3 tree level, ⟨n₁ n₂ n₁⟩ at (0, 0.4, 1),
  (0, 0.7, 0.7), (0, 0, 0.7), (0, 0, 0), (0.3, 0.3, 0.7), (0, −1e−6, −1e−6),
  (0, −2e−6, −1e−6), (0, 0.4, 0) and (0, 0.7, 0.7 + 1e−12): moves by
  ≤ 2.6e−10 absolute and ≤ 4.0e−9 relative per diagram (≤ 1.2e−10 and
  2.8e−9 grouped). Per-diagram and grouped Phase J now agree to 1.5e−15
  relative (before: 3.5e−9). The values of the k = 3 table above are
  unchanged at the digits shown.
- `single_population_quad_exp_test`, k = 2, ℓ = 1, ⟨n₁ n₂⟩: the one-loop
  term moves by ≤ 1.1e−9 absolute, ≤ 5.3e−6 relative per diagram (τ = 0:
  +2.016428e−4 → +2.016417e−4) and ≤ 2.2e−10, 1.1e−6 grouped; the total
  by ≤ 2.9e−7 relative. The tree level is unchanged. Per-diagram and
  grouped Phase J now agree to 1e−15 relative (before: 4.2e−6).
- `multipopulation_test`, k = 3 tree, ⟨n_E₁ n_E₂ n_E₁⟩, at the exact tie
  (0, 0.4, 0.4), the points around it (0, 0.4, 0.4 − h) for h = 1e−5,
  1e−6, 1e−7, and the tie (0, 0.7, 0.7). This configuration depends on the
  diagram cache:
  - with a fresh cache, only (0, 0.4, 0.4) reaches the fallback (16
    regions): +1.6859737e−6 → +1.6859242e−6, which is the tie-order limit
    to 5.9e−15 relative (before: 2.9e−5; see the known issue after the
    k = 3 table above). The other four points reach no fallback and are
    unchanged;
  - with the developer cache (other diagram representatives), every one
    of the five points sends 64 regions to the fallback, and every value
    moves: by 1.72e−4 relative at the tie and the three points around it
    (for example (0, 0.4, 0.4 − 1e−5): +1.6862178e−6 → +1.6859277e−6) and
    by 1.79e−4 at (0, 0.7, 0.7) (+2.4553191e−6 → +2.4548792e−6). The tie
    is the limit to 6.9e−15.

  The new values agree between the two caches to ≤ 1.6e−14 relative at
  all five points. With the developer cache, grouped Phase J moves by up
  to 2.7e−5 at these points, and the k = 3 ℓ = 1 value at (0, 0.4, 1) by
  1.1e−4 (64 fallback regions).
- `single_population_spike_reset_test`, k = 4 tree, ⟨n₁ n₂ n₁ n₂⟩ at the
  slice-1 point (0, 0.5, −1e−6, −1e−6): with the developer cache
  −4.82408912e−2 → −4.82408918e−2 (1.2e−8 relative; 1592 regions served
  by the fallback, 992 of them with three or more integration times, in
  the state the cache had then); with a fresh cache (1668 regions, 1088
  of them with three or more times) −4.8240891838e−2 → −4.8240891841e−2
  (7.6e−11 relative). The new values agree between a developer cache and
  a fresh one to 8.3e−15. At the nearby point (0, 0.5, −1e−6, −1.2e−6)
  (fresh cache; the total moves by 1.8e−9) the 192 regions that took a
  second pass (32 with two times, 160 with three) were compared one by
  one with 60-digit exact references: all within 3.9e−9 relative (thin
  regions; see the known limit below).

**Unchanged (measured, with a fresh diagram cache).** Models that make no
fallback call are bit-for-bit identical, compared with the old code in
one process with the new default: `ou_quartic` (k = 2 up to ℓ = 3, k = 4
ℓ = 1), `ou_quartic_colored`, `ou_quartic_two_dim_color_corr`,
`linear_hawkes`, `multipopulation_test` (k = 2; for k = 3 see above),
`single_population_linear_delta_spikes_test` (k = 2 and k = 3), and the
spatial models of the "Unchanged" list above. With other diagram
representatives a configuration can reach the fallback where a fresh
cache does not (as `multipopulation_test` k = 3 above); it then moves by
the old fallback's error.
With `NQUAD_HARDENED = False`, and under `DAEDALUS_PHASE_J_LEGACY=1`, the
fallback-served models above are bit-for-bit identical to the code before
this change.

**Cost.** Where the fallback is a small part of a run, the run takes about
as long as before. Where the fallback dominates, the cost depends on the
modes. With a few modes per edge the closed-form innermost time makes it
faster. With many modes per edge a per-diagram evaluation can be slower
than the old fallback, because the closed form expands the integrand
into one exponential per combination of modes of the edges that contain
that time: 16³ = 4096 terms per region for the `multipopulation_test`
k = 3 tree. Wall times, old and new code in one process, on a loaded
machine:

- the spike-reset model, k = 2 ℓ = 1 at 5 τ per diagram, build and
  evaluation: 6.2 s → 6.1 s (old and new passes interleaved, four each
  on a lightly loaded machine: 6.2–6.3 s and 5.9–6.4 s). The fallback's
  own time drops from 0.32 s to 0.07 s; the diagram build dominates;
- its k = 3 tree at the 9 points: 3.9 s → 1.6 s;
- its k = 4 tree at the slice-1 points (0, 0.5, −1e−6, −1e−6) and
  (0, 0.5, −1e−6, −1.2e−6), fresh cache (3336 regions, 2176 of them with
  three times), evaluation: 414 s → 259 s, measured before the last two
  changes below;
- `single_population_quad_exp_test` k = 2 ℓ = 1: 94 s → 94 s (the
  diagram build dominates);
- the `multipopulation_test` k = 3 tree at the five points above, with
  the developer cache (64 fallback regions per point), evaluation: per
  diagram 8.7 s → 11.1 s (1.27x), grouped 180 s → 12.2 s (pole tuples:
  65536 per region);
- its k = 3 ℓ = 1 point (0, 0.4, 1): 1.9 s → 2.8 s, measured before the
  last two changes below.

Two changes leave the values bit-for-bit identical, or move them only
where a cut moves, and cut the fallback's own time. First, the closed
form's per-call exponentials are combined by outer products with an
in-place series. Second, when only the outermost cut has to move, the
strip between the two cuts is integrated and added, instead of the whole
region again. On the regions captured from these runs, replayed in one
process, the time went from 11.6 s to 8.0 s for the 320
`multipopulation_test` regions, and from 46 s to 33 s for the 1792
spike-reset k = 4 regions. The values are bit-for-bit identical except
for the regions whose cut moved: 35 and 191 of them, by at most 2.1e−16
and 3.5e−15 relative. The totals of the public configurations are
unchanged at the digits shown above.

**Known limit.** Regions with many integration times stay expensive: the
quadrature still nests one adaptive rule per time beyond the innermost,
so its cost grows like (nodes per time)^(m − 1), now at the tighter
tolerance. For example `ou_quartic` k = 4 ℓ = 2 at μ = 1.25 sends 936
regions with up to 8 times to the fallback; one evaluation did not finish
in 15 min before this change, and none was attempted after it. (At
μ ≤ 1.1 no region reaches the fallback.) The tolerances and the
truncation follow the result, but each region's accuracy still rests on
QUADPACK's error estimates at every level, on the breakpoints and on the
decay certificate. Integrands that cancel heavily can end with QUADPACK's
roundoff warning (counted and warned about): a model-free region whose
modes oscillate with angular frequency 300 (e^{(−1 ± 300i)u}) ended
7.9e−10 relative off the exact value. A grouped integrand expands the products of modes into pole
tuples, so close poles cancel in the integrand itself: a model-free region
with residues ±2048 is accurate to 2.4e−10 grouped and 2.6e−13 per
diagram. Without known modes every time is integrated by quadrature, and
an oscillating region is slow: a model-free one with modes
e^{(−0.2 ± 25i)u} took 100 s on a loaded machine and was accurate to
9e−12, with QUADPACK roundoff warnings in its inner integrals (counted and
warned about). A thin region is limited by the rounding of its own
bounds and row values, to roughly (rounding unit of its coordinates) /
(its width): regions 2e−7 wide at the spike-reset k = 4 point
(0, 0.5, −1e−6, −1.2e−6) are accurate to 4e−9 (before: 1.9e−3), with
QUADPACK's roundoff warning on some of them; a model-free wedge with row
coefficients 1e−12 (not a Phase J shape), a few thousand rounding units
wide, to about 1e−5. Where the inner times' cuts move (rows other than ±1
ordering rows that are edges; no public model has them), the region is
integrated a second time. On regions whose accuracy is limited by such
rounding the nested quadrature chases the noise, and its cost then varies
erratically with the cut. A model-free region with m = 4 times, slices
8192 times wider than their distance to the end and modes that cancel to
2^−25 took 22–24 s in one pass with the cut at K = 40 or 52, but more than
5 minutes at K = 48 or 59. Its checked evaluation (inner K = 59) did not
finish in 15 minutes; before, it took 23 s and came out 2.4e−7 low, with
QUADPACK's roundoff warning. The ordered real and imaginary passes keep a
part that is only the other part's noise from driving the subdivision,
but a part limited by its own rounding is still refined to its
subinterval limit. The first pass's absolute tolerance rests on a
64-point sample of the integrand, floored at 1e−3 times its bound; a
sample that misses a peak can sit many orders below the integrand.

### Fixed: poset lower-bound inheritance; an exact integrator for regions with three or more times (numbers move)

The analytic integrator for regions with three or more integration times
(the poset integrator) splits a region into causal orderings and integrates
every time of an ordering from one shared lower bound L, the scalar lower
bound that some of the times have. A time without a lower bound of its own
is bounded below by L only if a time it must follow has one: s_v > s_u > L.
Before this change every time inherited L. A time with no such predecessor
extends below L, down to the integration box, and that part of the region
was cut off. This is the "poset lower-bound inheritance" error of the known
issues above.

- **The inheritance rule.** The poset integrator now accepts a shared L only
  when every time has a scalar lower bound of its own or a predecessor with
  one, in the transitive closure of the causal ordering. Otherwise it passes
  the region on. A region with no scalar lower bound at all is unchanged
  (it still uses the earliest external time − 50, a separate known limit).
- **An exact integrator for what the poset integrator passes on.** Every
  region with three or more times that the poset integrator passes on, for
  any reason (the rule above, unequal scalar lower bounds, ordering rows
  shifted by more than 1e−9, an ordering cycle of positive total shift, an
  overflowing or degenerate closed form), now goes to a new exact
  integrator, `engine/integration/time_domain/dbm_integral.py`, before the
  `scipy.nquad` fallback. It treats the region as a system of difference
  constraints (s_j − s_i < c and s_i ≷ c, the form every ordering and
  scalar row has), eliminates one time after another in closed form,
  splitting the region into the cases that decide which bound is the
  tightest, and keeps every exponential in logarithmic form until the end,
  so no intermediate factor overflows or underflows on its own. A direction
  that the rows leave open is closed at ±200, the box of the old
  default-tolerance `scipy.nquad` routines; a finite bound is never clipped.
  A cycle of total weight ≤ 0 (exact, no tolerance) makes the region empty
  or of zero measure: 0. A constant row follows the Θ(0) rule and the tie
  order of "Changed: Itô equal-time rule" above; every other row enters as
  it is (the value is continuous in its shift). The per-diagram and the
  grouped Phase J path use it alike.
- **What it does not take.** A region stays with the hardened `scipy.nquad`
  fallback, as before, and is counted, when a row is not a difference row
  (for example a ConvVertex kernel row with three times), when a term would
  overflow, or when the closed form is ill-conditioned: its estimated
  rounding error (1e−15 times the summed magnitudes of all its terms)
  exceeds 1e−10 of the value or 1e−14 of the integrand's scale. Close poles
  (pole sums that nearly cancel) and thin regions do that; the closed form
  is then not trusted. A term whose coefficient cancels to exactly 0 keeps
  its magnitude in that estimate. Before giving up, an ill-conditioned closed
  form is computed once more with every interval between two constant bounds
  that is thin for its exponent (|β| · width ≤ 1) integrated by 16-point
  Gauss–Legendre instead of an antiderivative difference that cancels; that
  value is kept only if it passes the same test, so every value accepted the
  first time is unchanged. This matters at τ = 0 of every k = 2 grid (the
  external time is moved to −1e−6, so some regions are 1e−6 thin): on
  `single_population_quad_exp_test` k = 2, ℓ = 1 the 16 such regions at
  τ = −1e−6 and 4 at τ = 2.5 are answered this way (the first closed form
  of the thin ones was 100 % off and was declined; the hardened fallback
  then did not finish the τ = −1e−6 point in 55 minutes, against 27 s with
  the flag off).
- **Known limits.** The ±200 box is absolute (measured from the time origin),
  as in the old routines, not relative to the region. Two consequences:
  - The hardened fallback truncates a direction with a decay certificate at
    40/κ beyond its last breakpoint, which is nearly box-free. For modes
    slower than κ ≈ 0.15, a region that went to the fallback before (any
    refusal other than the inheritance rule) can therefore move by the box
    truncation; no measured public configuration has such a region (none of
    them sent a region with three or more times to the fallback, before or
    after).
  - A region bounded above by an external time near or below −200, with a
    time open below, is truncated by the box or emptied (answered as exactly
    0), whatever the decay rate. On `single_population_spike_reset_test`
    k = 2, ℓ = 1 this moves the one-loop term at τ = −199 by 2.7 % and at
    τ = −250 by 1.6 % (terms of order 1e−59 and 1e−74), against the same
    regions with the box widened to ±10⁴; at τ = ±10 and ±150 no region
    moved by more than 1e−8 relative. The default τ grids stay inside ±50.

  The provenance stamp of saved results (`phase_j_convention`) does not
  record `USE_DBM_FALLBACK`.
- **Flag.** `USE_DBM_FALLBACK` (default `True`), environment variable
  `DAEDALUS_PHASE_J_DBM=1|0`; `False` restores the code before this change
  bit-for-bit (see "Added" below). Both changes apply only with the Itô
  rule: under `THETA0_CONST_ROW_MODE = 'legacy_clip'` the code before this
  change runs, so that mode keeps reproducing the pre-0.2.0 numbers.

**Moved (measured; current values, not validated results).** "Before" is
the code immediately before this change (it includes the hardened fallback,
which moved the spike-reset τ = 10 one-loop term in its sixth digit since
the table above was measured). Tree levels are unchanged.

`single_population_spike_reset_test`, parameters as in the tables above,
k = 2, `max_ell = 1`, ⟨n₁ n₂⟩(τ):

| τ | one-loop term, before | one-loop term, after | total, before | total, after |
|---|---|---|---|---|
| 0 | −5.06015e−3 | −5.06015e−3 (+5.6e−9) | +5.25929e−1 | +5.25929e−1 |
| 1 | −6.36836e−3 | −3.86596e−3 | +1.66995e−1 | +1.69498e−1 |
| 3 | −1.29531e−3 | −1.02615e−3 | +1.20710e−3 | +1.47626e−3 |
| 5 | +1.19179e−4 | +3.82909e−5 | −4.65523e−3 | −4.73611e−3 |
| 10 | +4.65706e−5 | +3.88692e−5 | −2.72079e−4 | −2.79781e−4 |

- At τ = 1 and 3, each of the 96 affected regions (of 520 with three or more
  times) agrees with tight quadrature on the same box to ≤ 3.7e−13
  relative (48 of them are empty there and exactly 0 on both sides); the new
  one-loop terms are the estimates of the known issue above (−3.866e−3 and
  −1.026e−3).
- Per-diagram and grouped Phase J agree to ≤ 8.5e−15 relative (one-loop
  term; 7.7e−15 with the flag off: rounding).
- `scipy.nquad` fallback calls are unchanged (40 per diagram set, 10
  grouped, all two-time regions).
- Evaluation time (the five τ, flag off vs on in one process, median of
  three alternating passes): 2.40 s and 2.64 s; the whole
  `compute_cumulants` run (warm cache), 15.4 s and 15.5 to 15.9 s.

`single_population_quad_exp_test` (parameters `P_SP` of
`tests/tools/phase_j_zoo_baseline.py`), k = 2, `max_ell = 1`, ⟨n₁ n₂⟩(τ):
96 region evaluations (over the three τ) are refused by the inheritance rule
and answered exactly (20 of them by the thin-interval retry above). Its
`scipy.nquad` fallback calls are unchanged (24, all two-time regions), and so
is its evaluation time (the three τ in one process: 75.8 s off, 75.5 s on).
Its regions with no scalar lower bound at all still use the margin of 50
(unchanged, the separate known limit above): with the margin at 200, the box
of the new integrator, the one-loop term at τ = 2.5 would be +3.89211e−4,
3 % higher.

| τ | one-loop term, before | one-loop term, after | total, before | total, after |
|---|---|---|---|---|
| 0 | +2.01642e−4 | +2.01642e−4 (unchanged) | +3.67409e−3 | +3.67409e−3 |
| 2.5 | +3.77432e−4 | +3.77782e−4 | +1.07405e−2 | +1.07409e−2 |
| 10 | +2.92006e−4 | +2.95594e−4 | +7.46875e−3 | +7.47234e−3 |

**Unchanged (measured, bit-for-bit in one process, flag on vs off):**
`ou_quartic` k = 2 with ℓ = 3, k = 3 and k = 4 with ℓ = 2;
`ou_quartic_colored` and `ou_quartic_two_dim_color_corr` (k = 2, ℓ = 1);
`linear_hawkes`, `multipopulation_test` and
`single_population_linear_delta_spikes_test` (k = 2, ℓ = 1);
`single_population_spike_reset_test` k = 1 with ℓ = 1 and ℓ = 2 (two-loop
−8.89854e−4). None of them has a region refused by the inheritance rule or
a region with three or more times at the fallback. Spatial models do not
use this integrator. `dendritic_quad_soma_sigmoid` and
`quadratic_hawkes_alpha` were not re-measured (over the time budget, see
"May move" above).

### Added

- **Prediagram cache: version stamp, four more shipped cells, manifest, explicit
  fetch.** The v2 prediagram cert files are now stamped (format, convention
  `prediagrams_v1`, edge-pack version, certificate backend, Sage version, cell,
  class count). A stamp from another convention is refused with a message
  naming both; unstamped legacy files are still accepted. The cells (2,4),
  (3,3), (5,2) and (6,1) now ship in git as xz-compressed stamped files
  (5.7 MB; class counts equal the v1 records and the independent recount
  logs), next to a `MANIFEST.json` (name, cell, class count, size, SHA-256 of
  every shipped file, checked by tests). The cells too big for git, (4,3) and
  (6,2), are downloaded by the explicit `dd.fetch_cache(cells, dest, base_url)`
  (base URL from `DAEDALUS_CACHE_URL`; SHA-256, size, stamp and class count are
  verified before the file is moved into place; never called implicitly).
  Numbers do not change.

- **Noise-source library (`api/noise.py`).** Every noise source is given by
  its per-unit-time cumulant generating function `K(theta; x)`,
  `E[exp(theta d eta) | past] = exp(K dt)`, and enters the MSR action as
  `-K(sum_i c_i psi_i)` (so `Gaussian(var='2*D')` gives `-D*xt^2` and
  `Poisson(rate='phi')` gives `-(exp(xt)-1)*phi`). Sources: `Gaussian(var,
  mean)`, `Poisson(rate, size)`, `CompoundPoisson(rate, jump)` with the jump
  laws `Dirac`, `Exponential`, `Gamma`, `GaussianJump`, `Laplace`, `Uniform`,
  `Bernoulli`, `Binomial`, `FinitePMF`, `GammaProcess`,
  `InverseGaussianProcess`, and the escape hatches `Cumulants([...])` and
  `CGF(expr)`. Parameters may depend on the fields; each source checks
  `K(0) = 0`, rejects an explicit time `t` and validates its symbols. New
  module only; no engine or model changes.
- **SDE front-end (`api/sde.py`).** `SDE(name)` declares fields, their
  equations in the operator form of `.equation` (`.equation('v',
  lhs='(tau*Dt + 1)*v', rhs='Em + w*g*n')`, or `.drift('x', f)` for
  `lhs='Dt*x'`) and noise sources from `api/noise.py`
  (`.noise('xi', Gaussian(var='2*D'), couples={'x': 1})`). It derives the
  MSR action `sum_x xt*(lhs - rhs) - sum_a K_a(sum_x c_xa*xt)` (wrapped in
  `sum(... for i in <population>)`) and the mean-field equations
  `lhs|_{Dt=0} = rhs + sum_a c_xa*K_a'(0)` (kernels replaced by their
  integral), and emits them through `TemporalModelBuilder`
  (`set_action_text` + `.equation`); `.to_builder()` returns that builder for
  further hand edits, `.show()` prints the derived calls. A source can be
  exposed as a field (`expose='n'`: the point-process form
  `nt*n - K(nt)` of the Hawkes models). Ito by default;
  `.interpretation('stratonovich')` adds the Gaussian drift correction and
  raises for jump or exposed sources. Populations, indexed parameters,
  functions and kernels pass through unchanged. New module only; no engine or
  model changes.
- **Call-time Phase J flags.** The flags are module attributes of
  `engine.integration.time_domain.final_integral`. Their environment variables
  are read once, at import. After that, the attribute is read at call time, so
  `monkeypatch.setattr` (or plain assignment) takes effect at once, also for
  results whose diagrams were built before the change. This applies to the
  grouped path too.
  - `THETA0_CONST_ROW_MODE`: `'ito'` (default) or `'legacy_clip'` (the pre-0.2.0
    rule). Set it with the environment variable
    `DAEDALUS_PHASE_J_THETA0_CONST_ROW=ito|legacy_clip`. An unknown value
    raises.
  - `POLYGON_BBOX_CAP` is now read when the analytic integrators are called
    (`bbox_cap=None` defaults), in every mode. Before 0.2.0 the default was
    bound at definition time, so changing the attribute had no effect.
  - `STRUCTURAL_ZEROS`: `True` (default) or `False` (the code before the
    change "Phase J skips identically-zero work" above). Set it with the
    environment variable `DAEDALUS_PHASE_J_STRUCTURAL_ZEROS=1|0` (also
    `true`/`false`, `yes`/`no`, `on`/`off`). An unknown value raises. Unlike
    the other flags, its δ-subset and pole-free decisions are made when a
    diagram is built, that is by the value in force when `compute_cumulants`
    runs (the skipped subsets are exactly 0, so this does not change
    values); the cycle test reads it at every call.
  - `NQUAD_HARDENED`: `True` (default) or `False` (the default-tolerance
    `scipy.nquad` fallback of the code before "Changed: hardened quadrature
    fallback" above, bit-for-bit). Set it with the environment variable
    `DAEDALUS_PHASE_J_NQUAD_HARDENED=1|0` (also `true`/`false`,
    `yes`/`no`, `on`/`off`). An unknown value raises. Read at every call.
    The fallback's settings are module attributes too: `NQUAD_EPSREL`
    (1e−10), `NQUAD_EPSABS_FACTOR` (1e−13), `NQUAD_LIMIT` (200),
    `NQUAD_LIMIT_MAX` (12800), `NQUAD_TAIL_K` (40) and
    `NQUAD_UNCERTIFIED_CAP` (200).
    `final_integral.nquad_warning_scope()` (a context manager) aggregates
    the fallback's warnings over everything evaluated inside it (see
    `PhaseJNquadFallbackWarning` below); the callables that
    `compute_cumulants` returns open one per call.
  - `USE_DBM_FALLBACK`: `True` (default) or `False` (the code before "Fixed:
    poset lower-bound inheritance …" above, bit-for-bit). Set it with the
    environment variable `DAEDALUS_PHASE_J_DBM=1|0` (also `true`/`false`,
    `yes`/`no`, `on`/`off`). An unknown value raises. Read at every call.
    It acts only while `THETA0_CONST_ROW_MODE` is `'ito'`.
- **`DAEDALUS_PHASE_J_LEGACY=1`.** This umbrella switch sets every Phase J
  flag to its pre-0.2.0 behaviour: the Θ(0) rule (`THETA0_CONST_ROW_MODE =
  'legacy_clip'`), `STRUCTURAL_ZEROS = False`, `NQUAD_HARDENED = False`
  and `USE_DBM_FALLBACK = False`. It reproduces the
  pre-change Phase J numbers bit-for-bit within one process, at the same
  external times and at the bounding box in force (so a pre-change run at
  another `POLYGON_BBOX_CAP` is reproduced by setting the attribute). It
  restores the Phase J rules only, not the times at which `dd.run` evaluates
  the points of its k ≥ 3 slices (see "Changed: `dd.run` k ≥ 3 slices …"
  above): with it set, the pre-0.2.0 τ = 0 slice point is
  `result['total_C'](0, -1e-6, ..., -1e-6)`. Set it in the environment
  before `daedalus` (or `api`) is imported. It overrides the per-flag
  variables.
- **Result provenance stamp.** `compute_cumulants` records the Phase J
  convention it computed with in `result['config']['phase_j_convention']`,
  and `api.save.save_npz` and `save_csv` write it together with
  `daedalus_version`. The convention is `theta0_ito_const_rows` by default, or
  `theta0_legacy_clip` when the legacy flags were in force during the
  computation. (A result dict without that entry is stamped with the
  convention in force when it is saved.)
- **`load_npz`.** `api.save.load_npz` (also `dd.load_npz`) reads a saved
  `.npz` back and checks the stamp. The stamp counts as stale when it is
  missing, older than 0.2.0, or names the legacy convention. In that case the
  loader issues a `StaleResultWarning`, once per file and process: results for
  models with instantaneous propagator parts may be stale and should be
  recomputed.
- **Divided-difference kernels for the Phase J exponential integrals (off by
  default; numbers unchanged).** New module
  `engine.integration.time_domain.expdd`. Every closed form Phase J
  evaluates is an integral of an exponential over a simplex, which is a
  divided difference of `exp` (Hermite–Genocchi): a triangle is
  |det| · exp[a₀, a₁, a₂] and a chain L < s₁ < … < s_m < U is
  T^m · exp[v₀, …, v_m]. Degenerate poles are repeated nodes, so no
  polynomial special case is needed. The kernels work in log domain
  (`log_dd_exp` returns (c, mantissa) with c = max Re zᵢ), so the only
  failure left is a result above e^700. It is reported as a status (`None`
  from the value functions), never as `inf` or `nan`. They do not bail on
  large exponents the way the legacy triangle guard (|Re| > 600) does.
  - Functions: `log_dd_exp`, `dd_exp`, `unit_triangle`, `triangle`,
    `chain_simplex` (also L = −∞, from the dominant eigenvector; a chain
    with a prefix sum Re Pⱼ ≤ 0 diverges and raises
    `DivergentIntegralError`), `chain_with_uppers` (the integral of
    `_chain_with_intermediate_uppers` by a transfer matrix with projections,
    without cut-tuple enumeration), each with a `*_log` variant returning
    (status, log scale, mantissa).
  - One source runs compiled by numba or as plain Python (numba missing, or
    `NUMBA_DISABLE_JIT=1`). The two agree to rounding, not bit for bit.
  - Nothing in Phase J calls the module by default. The only opt-in is
    `final_integral.CHAIN_UPPERS_KERNEL` (`'legacy'` by default, or
    `'transfer'`; environment `DAEDALUS_CHAIN_UPPERS_KERNEL=legacy|transfer`,
    any other value is an error; read at call time), which routes
    `_chain_with_intermediate_uppers` to `expdd.chain_with_uppers`. The
    selector is part of the chain-uppers memo key, so a toggle misses the
    memo. With `'legacy'` every result is bit-identical to before.
  - Flag `USE_PHASE_J_DD_KERNELS` (environment
    `DAEDALUS_PHASE_J_DD_KERNELS=1|0`, default 0, any other value is an
    error). It is reserved for routing the triangle, chain and grouped paths
    to `expdd` in a later change; nothing reads it yet.
    `DAEDALUS_PHASE_J_LEGACY` touches neither.
  - Measured against mpmath at 80 digits (`tests/test_expdd.py`). The rule:
    an error is accepted if it is ≤ 1e−11 of the value, or ≤ 1e−14 of the
    integral of the absolute value when the value itself cancels. Every
    case passes. Worst error relative to the value (and to the integral of
    the absolute value where that is the one that passes):

    | family | cases | worst |
    |---|---|---|
    | exp[z₀..z_n], n = 1..7, generic/repeated/clustered/fast/oscillatory | 184 | 8.1e−14 |
    | unit triangle J(p, q) | 400 | 5.0e−16 |
    | triangles on boxes up to 2000 wide (exponents to ~7e4) | 243 | 4.9e−12 |
    | chains, m = 2..7: generic, OU-degenerate, close pairs, fast poles, random | 247 | 8.3e−14 |
    | chains with a block sum δ = 0..0.1; 200 captured degenerate calls | 240 | 2.2e−14 |
    | chains with intermediate uppers | 60 | 1.3e−14 |
    | L = −∞ vs a far L | 60 | 1.3e−14 |
    | oscillatory nodes (Re ∈ [−1, 0], \|Im\| ∈ [10, 1000]), n ≤ 16, and chains | 104 | 6.9e−10 of the value (which cancels), 7.4e−17 of the absolute integral |
    | long exp[z₀..z_n], n = 8..30, clustered/spread; chains m = 8..16 | 66 | 9.0e−15 |
    | mixed magnitudes 1e−6..1e3, n ≤ 12 | 120 | 8.4e−13 |

    `_chain_with_intermediate_uppers` with `'transfer'` vs `'legacy'`: 2.4e−14
    on the 320 captured calls, ≤ 7.1e−15 on the end-to-end values of
    `ou_quartic` (k = 2, ℓ ≤ 2; k = 4, ℓ = 2) and
    `single_population_spike_reset_test` (k = 2, ℓ = 1, per-diagram and
    grouped), with the same memo hits and misses and about the same wall
    time.
- **Developer-facing.**
  - `final_integral._SUBSET_HOOK`, a per-subset observation hook. Its
    payload includes each constraint row's kind and the external legs behind
    the free times (`tie_ctx`).
  - New counters in `final_integral._RUNTIME_COUNTERS`:
    - quadrature fallback: `nquad_calls`, `nquad_fallback_by_reason`,
      `zero_normal_rows_seen`, `polytope_m0_direct`;
    - the Θ(0) rule: `theta0_const_empty`, `theta0_const_drop`,
      `theta0_tie`, `theta0_tie_ordered`, `theta0_subsets_pruned`;
    - empty regions: `polygon_zero_area`, `poset_empty_const`,
      `poset_empty_cycle`;
    - skipped identically-zero work: `forced_delta_pruned` (δ-subsets not
      built) and `polytope_empty_cycle` (cycles answered at the fallback);
    - the hardened quadrature fallback: `nquad_hardened_calls` (regions it
      served; also counted in `nquad_calls`), `nquad_hardened_empty`
      (regions found empty without quadrature),
      `nquad_hardened_empty_intervals` (intervals answered 0 without
      evaluating the integrand), `nquad_hardened_capped` (regions with an
      open direction truncated at 40/κ), `nquad_hardened_cap_span_max` (the
      largest such distance), `nquad_hardened_uncertified` (regions whose
      open direction had no decay certificate, from the modes or, for rows
      other than ±1 ordering rows that are edges, from linear programming,
      and used the old distance 200), `nquad_hardened_no_modes` (regions
      without known modes),
      `nquad_hardened_general_rows` (regions with a row that is not an
      ordering row: Fourier–Motzkin bounds and enumerated vertices),
      `nquad_hardened_fm_capped` (of those, regions whose elimination
      exceeded 4096 rows), `nquad_hardened_kinks_incomplete` (steps whose
      vertices were not enumerated: more than 20000 row subsets),
      `nquad_hardened_quad_flags` (quadrature calls of a region's final
      pass that still ended with a nonzero `ier` and an error estimate
      above their tolerance and, multiplied by the widths of the enclosing
      levels, above 1e−10 times the region's scale, or at the limit 12800;
      the value is still used), `nquad_hardened_quad_retries` (calls that
      reached their subinterval limit and were repeated with 12800),
      `nquad_hardened_reruns` (regions integrated again with the absolute
      tolerance from the result), `nquad_hardened_tail_widened` (regions
      whose outermost open direction was cut farther out, the strip
      between the cuts integrated and added, or whose inner times' cuts
      moved out),
      `nquad_hardened_inner_unchecked` (regions whose inner cuts could not
      be checked),
      `nquad_hardened_innermost_overflow` (closed-form innermost integrals
      that overflowed and were integrated numerically instead) and
      `nquad_hardened_innermost_capped` (regions whose closed-form
      innermost expansion would exceed 65536 terms);
    - the poset inheritance rule and the exact difference-constraint
      route: `poset_lower_not_inherited` (regions the poset integrator
      refused by the inheritance rule), `dbm_attempted`, `dbm_answered`,
      `dbm_empty` (answered 0: an empty or zero-measure region),
      `dbm_answered_p3` (answered after a refusal by the inheritance
      rule), `dbm_answered_thin_retry` (answered by the thin-interval
      retry), `dbm_declined_rows` (a row that is not a difference row),
      `dbm_declined_overflow`, `dbm_declined_ill_conditioned` (a closed
      form whose estimated rounding error exceeds its tolerance),
      `dbm_declined_other` and `dbm_error_ratio_max` (the largest ratio of
      an answered region's estimated rounding error to its tolerance, ≤ 1).
      A region answered by that route is reported to `_SUBSET_HOOK` with
      `path = 'dbm'`.
  - `PhaseJNquadFallbackWarning` (a `UserWarning`, in
    `engine.integration.time_domain.final_integral`). It is issued once per
    model, source (per-diagram or grouped Phase J) and loop order, per
    process, when the hardened fallback integrates regions there. The
    regions of one evaluation are aggregated: everything inside one call of
    a callable that `compute_cumulants` returns, or inside its own τ-grid
    evaluation. The warning gives the number of regions and of diagrams
    served, the regions without known modes and those without a decay
    certificate (with the weaker settings used there), and names the first
    diagram (loop order, external legs, edges, δ-subset). A second warning,
    with the same key and rule, gives the number of regions whose
    quadrature did not reach its tolerance (`nquad_hardened_quad_flags`)
    and the first QUADPACK message. Before, one warning was issued per
    typed diagram: a single `single_population_spike_reset_test` k = 4
    evaluation printed 396. The key uses the model's name, so building the
    same model again (another `compute_cumulants` call, other parameter
    values) does not repeat it, while another model does.
    (`compute_cumulants` passes the model's `name` to Phase J in
    `propagator_data['model_name']`; a model without a name, or a direct
    caller, is keyed by its `propagator_data` dict.) Outside an evaluation
    scope (direct calls of the integrators), and in forked workers of a
    parallel batch, the warning is issued at the first region. With the
    `logging` level at DEBUG, the module's logger names every diagram the
    fallback serves, once.
- **Model analyzer (`api/analyze.py`).** `analyze(model, question=None,
  policy=None)` reads a model dict from any builder (`TemporalModelBuilder`,
  `SpatialModelBuilder`, `api.sde.SDE`) and returns a `ModelReport`: frozen
  `ModelTraits` (field counts, polynomial or not, vertex species,
  state-dependent noise, undriven fields, time homogeneity, every saddle with
  its linear stability and marginal modes, the propagator class with its
  poles, repeated and close poles, fast/slow ratio, delta part and
  causality, and the noise class: white, colored rational, Markov-embedded,
  non-Gaussian, cross-correlated), typed `Finding`s (`code`, level
  `error|warning|info`, `message`, `remedy`) from gates G0-G4, and an
  assumptions ledger of the consents, conventions and assertions a result
  would depend on (Taylor truncation, Ito reading of state-dependent noise,
  Markov embedding, loop truncation, the chosen saddle among several stable
  ones, asserted positivity at the saddle). The user declares the question
  (`k`, external fields, `max_ell`, parameter point, `fixed_point_index`)
  and the policy (`strict` promotes warnings to errors; consents such as
  `allow_taylor_truncation`). It never raises for a bad model, leaves the
  model dict unchanged, runs no diagram enumeration or Phase J, reads the
  expand cache when one fits and never writes it, and is not wired into
  `compute_cumulants`. K(omega) is formed directly from the (1,1) sector
  (`Dt -> I*omega`) and inverted exactly over QQ[i][omega]; the symbolic
  inverse of `build_propagator` is not used. New module only.
- **Symbolic tree-level covariance in the Fourier domain
  (`api/symbolic_out.py`).** `tree_covariance(model)` builds the propagator
  as `compute_cumulants` stage 2 does (no diagram enumeration), reads the
  noise matrix `D` from the (2,0) sector of the action (`S` contains
  `-(1/2) xt D xt`; `Gaussian(var='2*D')` gives `D = 2 D`, Poisson gives
  `D = phi(v*)`) and returns `C0(omega) = G D G^dagger` as a matrix of Sage
  expressions, fully cancelled, in the symbolic parameters. Convention:
  `G(t) = (1/2 pi) int exp(i omega t) G(omega)` and
  `C_ab(tau) = <x_a(0) x_b(tau)> = (1/2 pi) int C0_ab(omega) exp(-i omega tau)`,
  the order of `compute_cumulants(external_fields=[a, b])`. Colored noise
  works through the pipeline's own Markov embedding (extra physical fields,
  white noise on them) and cross-correlated noise gives an off-diagonal `D`.
  Exports per entry: sympy, LaTeX, a numpy callable with named parameters,
  and a JSON string with the parameter list. `pair_with_kernel` returns
  `int d omega/(2 pi) conj(L~) C0` by residues (exact, when every irreducible
  factor of the denominator has degree <= 2 in omega; the half plane of each
  pole is checked over a parameter neighbourhood) with a quadrature fallback
  whose tolerance it reports; `inverse_transform` gives the exact `C(tau)`
  and the weight of the `delta(tau)` contact term. Spatial models,
  non-rational propagators and non-local (not Markov-embedded) noise raise
  specific errors. New module and tests only; no engine or model changes.

### Tests

- The frozen Phase J fixture `spike_reset_k2_ell1` moves with this release, as
  tabulated above (twice: the Θ(0) rule and the poset lower-bound fix). Its
  regression test is a strict expected failure until the fixture is
  refrozen.
- `tests/test_analyze.py` checks the analyzer: a golden traits table for every
  tracked model and three SDE-front-end references; a gallery of unsupported
  models (an undriven field, an unstable saddle, explicit time dependence, a
  delay and a square-root kernel, a missing external field, a marginal mode,
  two stable saddles), each giving its finding code without an exception;
  and, per model, under 10 s with the model dict unchanged (the two models
  whose cold expansion exceeds that are timed when their expand cache is
  warm; the cold run of the heaviest is a slow test).
- `tests/test_dbm_integrator.py` checks the exact difference-constraint
  integrator: it equals the closed-form chain simplex on a plain ordering;
  shifted orderings, unequal lower bounds, a time bounded only by the
  integration box and a thin strip agree with tight nested quadrature; an
  ordering cycle, a pair of rows with Δt ≡ 0 and a constant row with value
  0 give exactly 0; terms that would underflow or overflow on their own are
  kept in logarithmic form, and a genuinely astronomical value is reported
  as an overflow (the region goes to the fallback). Regions that need the
  case split (a hexagon, a three-time band region) agree with quadrature;
  near-zero exponents on the ±200 box, a thin region and a closed form that
  cancels to exactly 0 are exact or declined; a thin constant-bound interval
  is answered by the retry; the box closes open directions only, at the
  default `_nquad_outer_cap()`.
  `tests/test_phase_j_p3_inheritance.py` checks the inheritance rule on
  hand-built orderings, a model-free region with three times taken from the
  spike-reset model (its exact value 2.60362113647539e−7; the poset
  integrator gave 2.7212e−9 before the fix), a constant row at exactly
  coincident external times (the tie order), and, end to end on
  `single_population_spike_reset_test` (k = 2, ℓ = 1), that every refused
  region is answered by the new integrator, that the one-loop term at
  (0, 1) moves from −6.36836e−3 to −3.86596e−3 (at (0, 3) from −1.29531e−3
  to −1.02615e−3), that grouped and per-diagram Phase J agree to 1e−12, that
  every region with three or more times forced through the new integrator
  gives the same total, and that the value at the exact tie (0, 0) is the
  left limit, with the tie order reaching the new integrator.
- `test_causal_poset.py::test_consistent_scalar_lower_missing_var` asserted
  the inheritance error itself (a time with no ordering edge inheriting the
  shared lower bound); it now gives that time a lower-bounded predecessor,
  and `test_consistent_scalar_lower_not_inherited` checks the refusal.
- The pre-change fixtures and the zoo baseline are kept unchanged under
  `tests/phase_j_refactor_fixtures/legacy/` and
  `tests/fixtures/phase_j_legacy_baseline.*`. The legacy-flag tests check
  against them. The four fixture copies are compared at their own
  tolerance (1e−10 relative) in every checkout. The zoo baseline was
  recorded with local diagram caches; from a fresh clone, zoo entries that
  reach `scipy.nquad` are compared at 1e−8 relative and without their
  route counters, because other diagram representatives take other
  quadrature routes. With the current defaults, in a zoo entry served by
  the hardened quadrature fallback (`single_population_quad_exp_test`,
  k = 2, ℓ = 1) the loop orders ≥ 1 are compared at 2e−5 relative and the
  total at 1e−6 (the old fallback's errors moved them by 5.3e−6 and
  2.9e−7); its tree level keeps the strict tolerance. That entry moves
  with "Fixed: poset lower-bound inheritance …" above, so its default-mode
  comparison runs with `USE_DBM_FALLBACK = False` (the other defaults
  unchanged). Under the legacy flags the comparison is unchanged.
- The exact-tie test of `multipopulation_test` (k = 3 tree at
  (0, 0.4, 0.4)), a strict expected failure since the tie order was
  introduced, passes with the hardened fallback.
- Tests no longer pick a diagram by its position in a list.
  `tests/_diagram_order.py` selects by structure (`_pick_live`) and breaks
  ties by `diagram_signature`. The `shuffle_diagram_order` fixture serves one
  test's diagram lists in a shuffled order: the prediagram records and, for
  each loop order, the typed diagrams with their multiplicities, so the
  diagrams of one prediagram are reordered among themselves too.
  `DAEDALUS_TEST_DIAGRAM_ORDER=shuffle` does so for a whole run (`reverse`
  reverses each loop order's typed diagrams). This is a probe, not a proof:
  one permutation can leave a positional pick in place.
  `tests/test_prediagram_cache.py`, which pins the exact record order, opts
  out.
- `test_bessel_two_c_line_bubble_matches_grid` (slow) is a strict expected
  failure: the `bessel` integrator misses the `grid` value on the KPZ
  two-correlation-line bubble. The test records the measured values.
- `test_bessel_matches_grid_at_unequal_external_times` is a strict expected
  failure that records the `bessel` bias at τ ≠ 0.
- `test_temporal_streamed_totals_match_eager_on_a_fallback_model` (slow)
  checks streamed against eager records on each model that reaches
  `scipy.nquad` (`single_population_spike_reset_test` per-diagram and
  grouped, `single_population_quad_exp_test`). With the fallback of commit
  d68e383 the spike-reset cases pass (rtol 1e-13) and `quad_exp` is a strict
  expected failure (1.21e−11 relative, measured 2026-10-06); a gap above
  1e−6 relative fails a case outright. It gates
  `TEMPORAL_CACHE_OFF_STREAMS`: run it after any change to the Phase J
  fallback, and flip the flag only when every case passes.
- The tests that need the Sage backend to re-derive the shipped
  certificates byte for byte skip on an installation whose canonical
  labelling does not (for example, Sage without bliss).

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
    integrator than the points around it, and two of those routes have
    known errors, the `scipy.nquad` fallback (default tolerance) and, for
    regions with three or more integration times, the poset lower-bound
    inheritance error (see the known issue under the k = 2 table below).
    A value at a tie can then be much less accurate than the values around
    it: on the default k = 4 `dd.run` slices of
    `single_population_spike_reset_test`, 1.0% to 8.5% relative (about
    1e−3 absolute) at τ = 0.5. See the known issue after the k = 3 table
    below.
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
- **Known issue: these one-loop values are not converged yet.** They still
  contain a separate, known error of the multi-time integrator (three or
  more integration times), scheduled to be fixed in a later release: in a
  causal ordering, a time variable can inherit a lower integration bound
  it should not have (poset lower-bound inheritance). At these parameters
  the error is large away from τ = 0. Measured against tight quadrature on
  every affected integration region (96 of the 520 multi-time regions),
  the one-loop term should be −3.866e−3 at τ = 1 (not −6.368e−3, an error
  of +2.50e−3) and −1.026e−3 at τ = 3 (not −1.295e−3, +2.69e−4; the total
  there is then about +1.48e−3, so the sign flip stands). At the τ = 0
  grid point the error is below 1e−8 absolute; τ = 5 and 10 were not
  measured. Expect these values to move again; the frozen fixture stays a
  strict expected failure until then.
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
  two of those routes have known errors:
  - Two-time regions can go to the `scipy.nquad` fallback and carry its
    error. For `multipopulation_test` ⟨n_E₁ n_E₂ n_E₁⟩ at tree level
    (parameters as above), (0, 0.4, 0.4) sends 16 such regions there (none
    at (0, 0.4, 0.4 − 1e−7) or at (0, 0.7, 0.7)). Its value, +1.68597e−6,
    is 2.9e−5 relative from the limit, +1.68592e−6: twelve times the jump
    between the two one-sided limits there (2.4e−6), so it is neither
    limit. The ties (0, t, t) of the same model at t = −2, −1, −0.5, 0.2,
    0.5, 0.7, 1 and 2 reach no fallback and equal the limit to ≤ 1.4e−14.
  - At an exact tie, a region with three or more integration times can be
    accepted by the analytic poset integrator, which rejects it (and sends
    it to the fallback) when the tied times differ by more than 1e−9.
    Accepted, it carries the poset lower-bound inheritance error (the
    known issue under the k = 2 table above): two tied external lower
    bounds count as consistent, and a time variable that has no lower
    bound of its own inherits theirs. The error is large. For
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
  evaluated at the times of slice 1). Fixes are planned in a later release: the hardening of the fallback and
  the poset lower-bound fix.

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
  gate. It has one strict expected failure per case:
  `single_population_spike_reset_test` per-diagram and grouped, and
  `single_population_quad_exp_test`. A case fails as an unexpected pass
  once its two totals agree to 1e−13 relative, and the flag flips only when
  every case does: the tight-tolerance agreement quoted above is rounding
  for `single_population_spike_reset_test` but about 1e−11 relative for
  `single_population_quad_exp_test`. Run the gate (`sage -python -m pytest
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

### Added

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
- **`DAEDALUS_PHASE_J_LEGACY=1`.** This umbrella switch sets every Phase J
  flag to its pre-0.2.0 behaviour: the Θ(0) rule (`THETA0_CONST_ROW_MODE =
  'legacy_clip'`) and `STRUCTURAL_ZEROS = False`. It reproduces the
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
      built) and `polytope_empty_cycle` (cycles answered at the fallback).

### Tests

- The frozen Phase J fixture `spike_reset_k2_ell1` moves with this release, as
  tabulated above. Its regression test is a strict expected failure until the
  fixture is refrozen.
- The pre-change fixtures and the zoo baseline are kept unchanged under
  `tests/phase_j_refactor_fixtures/legacy/` and
  `tests/fixtures/phase_j_legacy_baseline.*`. The legacy-flag tests check
  against them. The four fixture copies are compared at their own
  tolerance (1e−10 relative) in every checkout. The zoo baseline was
  recorded with local diagram caches; from a fresh clone, zoo entries that
  reach `scipy.nquad` are compared at 1e−8 relative and without their
  route counters, because other diagram representatives take other
  quadrature routes.
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
  records the gap between streamed and eager records on each model that
  reaches `scipy.nquad` (`single_population_spike_reset_test` per-diagram
  and grouped, `single_population_quad_exp_test`; measured with the
  fallback of commit 16b5564), as one strict expected failure per case; a
  gap above 1e−6 relative fails a case outright. It gates
  `TEMPORAL_CACHE_OFF_STREAMS`: run it after any change to the Phase J
  fallback, and flip the flag only when every case passes.
- The tests that need the Sage backend to re-derive the shipped
  certificates byte for byte skip on an installation whose canonical
  labelling does not (for example, Sage without bliss).

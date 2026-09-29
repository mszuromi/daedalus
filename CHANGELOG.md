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
    limit.
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
  curves move only through the first case. The k ≥ 3 curves of `dd.run`
  (`C_tau` and `C_tau_slices` of a k ≥ 3 result) do evaluate coincident
  times. With `kpoint_base_lags` left at its default, every non-swept
  non-anchor leg sits at −1e−6, so:
  - for k = 3, the τ = 0 point of each slice has coincident legs;
  - for k ≥ 4, EVERY point of a slice has at least two coincident legs (the
    k − 2 non-swept ones), so a whole slice curve can move;
  - the full grid (`kpoint_full_grid`, `C_tau_grid`) has coincident legs on
    its diagonals.

  A point with coincident legs moves only where the two one-sided limits
  differ, for example legs of different fields of a model with
  instantaneous parts (tables below). With distinct, nonzero
  `kpoint_base_lags` the non-swept legs stay apart, and the swept leg meets
  another leg only at a τ equal to one of the base lags.

Models without instantaneous parts (in every re-run: the OU family and the
spatial models) have no constant rows. Their numbers are unchanged at every
point, including coincident and nearly coincident external times. In
general, the new rules can change a value only where the evaluation decides
a constant row, skips a τ-independent empty subset, or applies one of the
exact emptiness tests above. When the counters `theta0_const_empty`,
`theta0_const_drop`, `theta0_subsets_pruned`, `polygon_zero_area` and
`poset_empty_cycle` all stay 0, the value is identical to the old one.
(`zero_normal_rows_seen` alone is not enough: it counts constant rows only
in the two-time and multi-time integrators.)

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
h = 1e−4, 1e−5, 1e−6; at (0, −1e−6, −1e−6) from h = 1e−7, 1e−8, 1e−9, so that
the samples stay below t₀ = 0). (0, −1e−6, −1e−6) is the τ = 0 point of the
`dd.run` k = 3 curve.

| model | (t₀, t₁, t₂) | before | after | limit | other limit |
|---|---|---|---|---|---|
| `single_population_spike_reset_test` | (0, 0.7, 0.7) | −4.58858e−4 | −9.89874e−2 | −9.89874e−2 | +4.12483e−2 |
| | (0.3, 0.3, 0.7) | +1.02854e−1 | −2.13653e−1 | −2.13653e−1 | +4.09230e−2 |
| | (0, −1e−6, −1e−6) | +1.01360e−1 | +4.96369e−2 | +4.96369e−2 | −3.32555e−1 |
| | (0, 0.4, 0) | +2.25093e−2 | −1.43972e−1 | −1.43972e−1 | (same) |
| `single_population_linear_delta_spikes_test`, `Em = [0.8, 0.78]`, `tau = [10, 9]`, `w = [[0, 0.25], [0.2, 0]]` | (0, 0.7, 0.7) | +8.07043e−5 | +1.56695e−4 | +1.56695e−4 | +6.34919e−4 |
| | (0.3, 0.3, 0.7) | +2.56166e−5 | +6.40873e−4 | +6.40873e−4 | +6.59190e−4 |
| | (0, −1e−6, −1e−6) | +2.64260e−5 | +6.80526e−4 | +6.80526e−4 | +6.60666e−4 |
| | (0, 0.4, 0) | +1.48826e−4 | +1.48826e−4 | +1.48826e−4 | (same) |
| | (0, 0, 0) | +1.06073e−5 | +6.80526e−4 | +6.80526e−4 (t₂ < t₁ < t₀) | |
| `multipopulation_test` (parameters of `tests/tools/phase_j_zoo_baseline.py`, `P_MP`), ⟨n_E₁ n_E₂ n_E₁⟩ | (0, 0.7, 0.7) | +2.4548779e−6 | +2.4548792e−6 | +2.4548792e−6 | +2.4548736e−6 |
| | (0.3, 0.3, 0.7) | +1.3250094e−6 | +1.3250003e−6 | +1.3250003e−6 | +1.3250139e−6 |
| | (0, −1e−6, −1e−6) | +9.6992059e−7 | +9.6992101e−7 | +9.6992101e−7 | +9.6991936e−7 |
| | (0, 0, 0) | +9.7179215e−7 | +9.6992006e−7 | | |

- The new values equal the limit to ≤ 1e−15 relative (linear delta spikes),
  ≤ 6e−14 (multipopulation) and ≤ 7e−11 (spike reset, whose k = 3 tree
  level is partly evaluated by `scipy.nquad`).
- The same k = 3 evaluations at distinct times, including times 1 ulp,
  1e−13, 1e−12 and 3e−12 apart and times shifted by 10⁶, are unchanged
  (bit-for-bit, in process). So is the k = 4 point (0, 0.3, 0.3 + 1e−12, 0.9)
  of linear delta spikes.
- At coincident times, `linear_hawkes` k = 3 (ℓ ≤ 1) values change by
  ≤ 3.1e−14 relative and `single_population_quad_exp_test` k = 3 tree values
  by ≤ 1.7e−14.

k = 4, tree level, ⟨n₁(t₀) n₂(t₁) n₁(t₂) n₂(t₃)⟩ of
`single_population_linear_delta_spikes_test` (parameters as above), at the
points of the default `dd.run` k = 4 slices: the swept leg at τ, the other
non-anchor legs at −1e−6. "limit" is the one-sided limit in which the
later-listed of the coincident legs approaches from below, "other limit"
the other one (Richardson extrapolation from h = 1e−7, 1e−8, 1e−9).

| swept leg | (t₀, t₁, t₂, t₃) | before | after | limit | other limit |
|---|---|---|---|---|---|
| 1 | (0, 0.5, −1e−6, −1e−6) | +1.79275e−6 | +3.00275e−5 | +3.00275e−5 | +2.97737e−5 |
| 1 | (0, 1, −1e−6, −1e−6) | +1.71067e−6 | +2.85880e−5 | +2.85880e−5 | +2.83773e−5 |
| 1 | (0, 2, −1e−6, −1e−6) | +1.55813e−6 | +2.59229e−5 | +2.59229e−5 | +2.57873e−5 |
| 3 | (0, −1e−6, −1e−6, 0.5) | +1.79275e−6 | +2.97737e−5 | +2.97737e−5 | +3.00275e−5 |
| 2 | (0, −1e−6, 0.5, −1e−6) | +6.96331e−6 | +6.96331e−6 (≤ 1.5e−15 relative change) | (same) | (same) |
| any, τ = 0 | (0, −1e−6, −1e−6, −1e−6) | +1.23614e−6 | +3.27274e−5 | +3.27274e−5 | |

- Every point of the slices whose non-swept legs belong to different fields
  (swept leg 1 or 3) moves, by a factor of about 17. Slices 1 and 3, equal
  before, now take the two different one-sided limits. Slice 2, whose
  non-swept legs belong to the same field, changes only by rounding
  (≤ 1.5e−15 relative) away from τ = 0.
- The new values equal the limit to ≤ 7e−16 relative.

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
coincident external times, moves as described above. Models outside the
public model set are not listed.

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
- **`DAEDALUS_PHASE_J_LEGACY=1`.** This umbrella switch sets every Phase J
  flag to its pre-0.2.0 behaviour: currently the Θ(0) rule. It reproduces the
  pre-change numbers bit-for-bit within one process, at the bounding box in
  force (so a pre-change run at another `POLYGON_BBOX_CAP` is reproduced by
  setting the attribute). Set it in the environment before `daedalus` (or
  `api`) is imported. It overrides the per-flag variables.
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
      `poset_empty_cycle`.

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

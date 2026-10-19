#### Pole engine: unit-invariant thresholds (audit wave W4b)

The pole engine (`DAEDALUS_PHASE_J_ENGINE=poles`, the supported backup of the state-space default) had absolute
thresholds that changed answers under an exact change of time or field units. Each is now relative to the problem's
own scale, and each fix is tested with exact time-rescaled (or field-rescaled) twins, units changed by 1e-6 to 1e6,
against the state-space engine or a closed form (`tests/test_audit_w4b_pole_units.py`).
Zoo A/B on BOTH engines (`use_cache=False`, base = main c8b869c): default engine 17 of 18 runnable public entries
bit-identical, `dendritic_quad_soma_sigmoid` within 2.8e-16 (main's own scatter); pole engine 15 of 16 bit-identical,
`spike_reset-k2-l1-grouped` within 4.1e-15 (main's own cross-process scatter on that route: repeated main runs differ
by as much); analyzer codes identical except `ou_quartic_two_dim_color_corr` on the pole engine, which now reports
`info:G6_ILL_CONDITIONED` (13 subset sums above their unit-correct floor, worst ratio 2.76; values bit-identical,
`G6_ACCURACY` unchanged at 6.1e-13). No zoo number moves beyond 1e-15 except that scatter.

- **Kernels with cos / sin / cosh / sinh** (`_extract_exp_modes`): `r e^{-rt} cos(rt)` at `r = 2000` was fitted as
  ONE real exponential, because the fixed probe times 0.37, 1.9 and 4.3 underflowed and compared nothing; the one-loop
  C was 39 % off with no warning (1.532e-3 instead of 1.105e-3 at τ = 0). At `r = 2` the same kernel was refused
  ("Phase J could not evaluate 3 diagrams", one-loop C = 0). The probe times are now multiples of the kernel's own
  time unit, and trig factors are written as exponentials first, so the kernel splits into its complex modes. Both
  twins now give the state-space value (1.10487e-3, 9.6889e-4, 1.8371e-4 at τ = 0, 0.5, 2; agreement 1e-15). A trig
  kernel that does not decay (`cos(ωt)` alone) is still refused. The state-space engine gets the same fix where it
  used this function (a mixed cos / sin / cosh kernel at rates 1e-4 was refused there too).
- **Residues from the numerical tiers** (when the exact tier fails): a residue entry below 1e-12 of the LARGEST entry
  at its pole was set to 0. A field in small units (`x' = -x + c z` with `c = 1e-14`) lost its genuine `C_xz`
  (pole-engine value 0 instead of 1.33e-14), and `propagator['C_mats']` was wrong on every engine. Entries are now
  zero only when they are structural zeros of `K^{-1}` (exact), or rounding noise relative to the SAME entry's size
  on a small circle around the pole (a quarter of the distance to the nearest other zero, at most half the pole's
  modulus: a circle as wide as a far zero clipped a genuine residue, CI). Residues now agree with the exact tier
  entry by entry (1e-12).
- **The numerical tiers are no longer entered silently**: an exact-tier inverse that raised or ran out of time, or a
  repeated root the cluster route did not resolve, now gives a `PoleResidueWarning`. When the exact pole record
  exists, the analyzer reports `G3_RESIDUES_NUMERICAL_TIER`.
- **Poles from the numerical tiers**: the retarded floor (`Im ω > 1e-9`), the pre-Newton dedup radius (1e-5), the
  Newton exits and steps (`|det| < 1e-13`, `1e-7 (1 + |ω|)`) and the probe frequencies were absolute. In slow units
  (`a = 1e-4`, `b = 1.05e-4`) two poles were merged, and `C_xx` came out 236 times too large with no warning. Every
  root of `det K(ω)` is now Newton-refined with relative steps and kept only if `det K` vanishes there. A zero off the
  real axis (|Im| > 1e-3 |ω|) keeps its sign; one near it is a retarded pole when `Im ω` exceeds 100 times its OWN
  first-order rounding error (from the singular vectors of `K(ω)`: invariant under scaling and non-normality),
  never a threshold set by other roots (one Newton routine; the three drifted copies are gone). After a
  leading-coefficient trim drops genuine poles, the untrimmed polynomial's roots are the candidates (one per root;
  restored and trimmed copies of an ill-conditioned slow pole could otherwise both be kept).
- **Retarded poles of the exact tier and of the cluster analysis** (`api._pole_clusters`): a decay rate below the
  absolute 1e-9 was dropped silently. Example: `y' = -b y` with `b = 5e-10` next to a unit-rate field, after
  downgrading `G2_MARGINAL_MODE`, gave `C_yy = 0` instead of 0.2. Its 1000-times-faster twin kept the pole. The
  decision is now made per root: a root clearly off the real axis keeps its sign; when a root is within 1e-3 of the
  axis (relative), or 1e3 times smaller than the largest, all roots of the exact `Q(ω)` are computed at 60 digits,
  matched one to one, and the near ones decided there (an exact zero root via `Q(0) = 0`); a root that cannot be
  matched sends the propagator to the exact 50-digit route. The cluster analysis decides per root at its 50 digits
  (exact zero roots from the exact factors). A
  first version of this fix used a floor relative to the LARGEST root, which dropped the pole of a model whose
  float parameters leave a genuine far root (a cancelling mass matrix: root ~1e16 times the pole) -- on the exact
  tier on every platform, and on the numerical tier where that far root verifies (CI, Linux); both are tests now.
- **Kernels written as complex exponentials** (`e^{(-r ± i r) t}` with `r` positive): the structural transform
  refused the complex rate `(1 ∓ i) r` as "not known to be positive"; the REAL part is what must be positive (the
  convergence condition), and is now checked. Where Sage's own integral of such a kernel fails (seen on the Linux
  conda stack), the kernel had no image and the mean-field check took it as normalised (gain 1 against the
  solver's 1/2), refusing a valid model with "saddle does not satisfy its own equations". Growing or undamped
  terms are still refused.
- **Poset chain simplex** (k ≥ 3, m ≥ 3 regions): scalar lower bounds within an absolute 1e-9 were treated as one. In
  units 1e9 times finer, the lowers 0 and 5e-10 of a region with upper 2e-9 were merged, and the subset came out
  36.6 % low (its unit twin went to the exact route). The tolerance is now 1e-12 of the poset's largest bound
  (`POSET_LOWER_RTOL`).
- **DBM / box-free conditioning floor** (`dbm_integral`): the absolute floor `COND_ATOL · scale` had the units of the
  integrand, not of the integral. The same integral changed from `ill_conditioned` to `ok` when time was rescaled
  (the ratio scaled as λ^-m). The pole route also passed `Σ|C|` without `e^{Re E}`, so an ill-conditioned closed
  form in a correlator tail (`e^{-30}`) was accepted with relative error 1.2e-6 and its mp retry skipped. The
  default scale is now `integral_scale = Σ|C| e^{Re E} · ∏_v min(w_v, 1/max|β|)`: the integrand times a volume in
  the fastest exponent's time unit, capped by the region's own extent `w_v`. The route uses it, and the DD routes'
  error accounting (counters for G6) uses the same volume. At rates of order one this is the old floor. One
  consequence: a single-variable cancellation between near-equal SLOW rates (`a = -1e-3`) is accepted in double
  precision, as its unit-rate twin always was, instead of going to the mp retry.

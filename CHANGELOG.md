# Changelog

This file starts at 0.2.0. The 0.2.0 entry covers the Phase J changes that
move numbers, and the tools that go with them. Other changes since 0.1.0 are
in the git history.

Entries that change numbers list every public model whose results moved, with
values before and after. "Before" is the code immediately before the change,
which still used the 0.1.0 rules; `DAEDALUS_PHASE_J_LEGACY=1` reproduces it.
"After" is the new default.

### Unreleased

#### Fixed: audit fix wave 1 -- core accuracy (F1 group 1)

Every fix has a regression test in `tests/test_audit_wave1.py`. Zoo A/B (11 public entries, separate processes,
warm-up discarded, `use_cache=False`, base main 4a9e932):
- 9 entries are bit-identical (values and analyzer codes);
- `quadratic_hawkes_alpha` moves 2.0e-16 and `spike_reset` k = 2 moves 1.4e-15, both inside main's own scatter on
  these entries (`spike_reset`: 2.6e-14 between main 4a9e932 and main 147f402, whose difference is spatial-only).

- **New opt-in `DAEDALUS_TAU0=exact`: k = 2 τ = 0 at the exact Itô left limit C(0⁻)** (`api.compute._tau0_mode`).
  The τ = 0 grid point (and `total_C(t, t)`, and the exact delayed tree's τ = 0) is sampled at τ = −1e-6, so its
  relative error is rate × 1e-6. That is unit dependent: 1e-3 at a rate of 1e3. With `exact` it is evaluated at equal
  times, where Phase J's exact tie rule (`final_integral._const_row_verdict`) gives the left limit.
  - Measured against independent mpmath references (the audit's oracle cases, all of ss / poles / grouped, ℓ ≤ 3):
    τ = 0 error ≤ 3.1e-14 (`nudge`: 4.6e-7 .. 3.0e-6). τ ≠ 0 points are unchanged.
  - The default stays `nudge`, so no number moves. Making `exact` the default is a pending decision; it moves every
    k = 2 τ = 0 value by its rate × 1e-6.
  - k ≥ 3 slices (`daedalus._kpoint_slice_times`, `me_output.slice_me`) keep the −1e-6 placement. The engine's tie
    order is that slice's left limit only for some leg orders.
- **Exact delayed tree is unit invariant** (`api._delay_exact`). The boundary-value problem's Jacobian was a finite
  difference `resid(e) - resid(0)`. It cancelled the O(N) constant (field scale squared): 8e-6 relative off at a field
  scale of 1e6, with no warning. Now:
  - the Jacobian is the exact linear part;
  - the state is solved in balanced coordinates, and re-solved in noise-scaled coordinates when the variances spread by
    more than 16.
  - Mixed units up to a 1e9 ratio used to be refused as singular or come out 4e-6 off; they now agree to ≤ 8e-15.
- **Several delays: the approximant's error is in G6_ACCURACY** (`api.delay._fold_delay_into_g6`). The Padé estimate
  `est_rel_error` (|C_n − C_2n|) is added to the accuracy estimate R, and G6_ACCURACY_LOW is raised above the threshold.
  Before, a run with a 6.5e-5 Padé tree error reported R = 2e-14. On a tree-only run, DelayOrderWarning now says "the
  tree" instead of "the loops".
- **A non-white `noise_structure` is refused.** Phase J never applies a temporal noise kernel. A legacy dict with
  `noise_structure={'temporal_type': 'colored', ...}` was computed as white noise *and* without the source amplitude
  (status ok, off by the factor D). The refusal comes from the analyzer (`G2_NOISE_STRUCTURE`, error) and from a hard
  stop in `compute_cumulants`. `classify_coefficient_factors` now always puts the constant source amplitude in the
  scalar prefactor; `source_time_info[v]['amplitude']` holds only what stays in the integrand.
- **gentreeg's exit status is checked** (`loop_diagram_enumeration._trees_of_order_fast`). A generator that died
  mid-stream left a silently truncated tree list. Fewer prediagrams were then written to a correctly stamped cache
  file. It now raises `RuntimeError` with gentreeg's stderr.

#### Spatial engine: k=2 completion, lag convention, total_C (audit wave 2)

Numbers move (silent-wrong before):

- **Two-loop k=2 spatial C(x, tau)**: the +-tau completion now follows the
  orbit-stabilizer rule (both leaf->time assignments / `external_wick_compensation`),
  not the external-leg kinds. {R,R} two-loop diagrams without a leaf-swap
  automorphism were counted once; they now get Gamma(tau)+Gamma(-tau).
  Before: the ell=2 coefficient of the cubic+quartic d=0 test model was 13 %
  high (0.153 vs Boltzmann 0.136) and C(+tau) != C(-tau) by 2 %. One loop and
  tree are unchanged (the two rules coincide there). Applies to the single-field
  driver and the coupled i==j branch.
- **Coupled spatial cross-correlators** (`external_fields=[a, b]`, a != b) now
  return the package convention C_ab(tau) = <a(0) b(tau)> at tree and loop
  level; before they returned <a(tau) b(0)> (C_ab(+tau) and C_ab(-tau)
  swapped). Auto-correlators are unchanged.
- **Spatial `total_C`** now reads the returned (tau_grid, spatial_grid) grid
  (loops included); `total_C_by_ell` holds the per-order increments. Before it
  returned the tree only at every max_ell. Off-grid points raise, except the
  single-field d=1 tree (exact closed form).


#### Changed: memory pre-pass is honest, `discard=` evaluates one diagram at a time, dressed lines are minimal (item 61)

- **Honest pre-pass** (`final_integral.estimate_modesum_plans`, `statespace.diagram_estimate`).
  - The exact-zero-prefactor test now also drops diagrams charged the state-space estimate.
  - A conductance-kernel line is charged as Phase J builds it. Dressed: one edge, the cascade's states, no δ part.
    Leg path: an extra vertex, the propagator edge with its δ part, then the kernel edge.
  - Each δ-subset is counted on its contracted DAG. Each placement of the kernel slots counts once.
  - Against an exact census of the built stationary states the estimate is now within 0.5 %. Examples: delayed-alpha
    spike-only Hawkes, 1 neuron, 12.89 vs 12.87 MB (was 185 MB); its two loops, 366.09 vs 366.08 GB; 2 neurons with
    `w_ii = 0`, 0.390 vs 0.389 GB (was refused at ~122 GB; now 4.9 GB at the default budget, under it, and 0.39 GB
    whenever the zero test runs).
  - The pre-pass of that run takes 2 s (kernel data cached per model).
- **Forked τ batches stay within the budget** (`pipeline._workers_within_budget`). Each worker may rebuild the lazily
  built states, so a run that has charged `used` bytes forks at most `budget / used - 1` workers, or runs serially.
- **`compute_cumulants(..., discard=False | True | 'auto')`.** Each diagram is evaluated on the τ grid right after it is
  built, its values are kept, and its plans and stationary states are released and refunded to the budget.
  - Values are bit-identical.
  - The pre-pass compares the largest diagram plus one segment working set with the budget, instead of the sum.
  - `'auto'` is on exactly when the sum is over the budget and that fits.
  - It needs fixed evaluation points: k <= 2 with a τ grid, no switch-on, not grouped, and a single pass.
  - The result's callables answer the grid points only. Elsewhere they raise `DiscardedDiagramsError` (exported by
    `daedalus`).
  - A diagram with shot-noise δ records keeps its charge (the pre-pass adds those diagrams to the peak).
  - Example: the Erlang-8 delayed negative feedback CME at k = 2, two loops sums 10.8 GB of stationary states (refused
    at the default budget); with `discard=True` the largest diagram is 225 MB and the run peaks at 2.8 GB resident.
- **Minimal dressed lines** (`statespace._ModelOps.dressed_entry`). The cascade propagator entry + kernel is
  balanced-truncated at the truncation tolerance in effect, with a Hurwitz check and the bound recorded in
  `Realization.truncation['dressed']`.
  - When the propagator cancels the kernel's poles, `n_G + n_K` states go down to `n_G`. Examples:
    - a 1-population exponential-kernel spike Hawkes at two loops goes from 10.6 to 0.62 MB of states and 71 to
      9.7 s, with values moving 1.3e-16;
    - the delayed negative feedback CME (hard delay, (n, n) Pade 4, item 56's direct terms) at k = 1, three loops
      goes from 703 to 27 MB and 557 to 96 s, with values moving <= 8.4e-15 (inside the 1e-12 truncation tolerance).
  - A rank-1 multi-population network keeps its cascade (nothing cancels).
  - Zoo: only `conductance` (k = 2, one loop) builds dressed lines; its values move by <= 2.0e-16. No other entry
    builds a dressed line.

#### Fixed: known-bugs sweep (item 60)

Numbers move only where they were wrong: a conductance vertex mixing a plain and a kernel term (bug 14, below).
Zoo A/B (11 public entries, separate processes, warm-up discarded): bit-identical except `spike_reset` k = 2
(1.4e-15 relative, inside main's own process-to-process scatter of 2.6e-14 on that entry) and
`dendritic_quad_soma_sigmoid` (1.4e-16, main scatters by the same). Every fix has a regression test in
`tests/test_known_bugs_60.py` (bug 2's are in `tests/test_leftovers_39.py`).
- **Silent wrong one loop for a conductance vertex mixing a plain and a kernel term (bug 14).** A vertex
  coefficient such as `c*z_h + ee` (drift `-ee*x*x - c*x*Conv(h, x)`) got its kernel slots counted on the whole
  coefficient and `z_h -> 1`, so the plain term was convolved too: one loop 0.49 relative off the Markov form (two
  kernels on one vertex: 0.86; a cubic mixed vertex: 0.64; a point-process model: 0.043). The diagram is now
  integrated term by term (one kernel monomial per vertex, its own slots and prefactor); all four agree with
  their Markov forms to <= 6e-14. A prefactor that is not the product of the vertex coefficients is refused by
  name. Single-monomial vertices are unchanged.
- **Derivative atoms in the Taylor expansion (bug 2).** Maxima's multivariate `taylor()` stops `k` orders early on a
  `k`-th derivative of an abstract function: `taylor(x*D[0](f)(x), (x, 0), 3)` has no `x^3` term. The per-atom
  expansion now takes such an atom of one field variable through the univariate form, which is exact. An atom of
  several field variables, or the legacy whole-action route (`DAEDALUS_EXPAND_LEGACY`, or an atom without a
  polynomial expansion) when the action has a derivative atom, raises `DerivativeAtomTaylorError` instead of
  returning a truncated expansion. No parsed model contains such atoms; a hand-built model dict could.
- **G6 sees the realization (bug 5).** A graded or not numerically Hurwitz rounded realization was already checked
  a posteriori and refused when broken (item 41). Its measured relative error (stationary variances by Lyapunov vs
  the same variances by adaptive frequency quadrature of the transfer function) now enters the state-space
  accuracy estimate R above 1e-6 (`result['phase_j_engine']['realization_check']`, analyzer
  `realization_err_rel`; below 1e-6 the frequency quadrature's own noise on strongly graded spectra dominates).
  A realization that passes the 1e-2 refusal check but is off by more than 1e-6 gives `G6_ACCURACY_LOW`. Graded
  Pade-8 / 24 realizations measure ~2e-16: R unchanged.
- **Standalone `analyze()` (bug 7)** makes the state-space route prediction itself, so its G3 pole-cluster
  findings get the in-run levels (e.g. `G3_POLE_CLUSTER_UNRESOLVED` is info, not warning, when the route covers
  the run).
- **Population named `I` in the simulator (bug 8).** `I` iterates as the population and computes as the
  imaginary unit (before: `range * float` TypeError in an expression using `I`).
- **`dd.Config(delay=...)` and `question.delay` in config files (bug 10)** pass the delay route's options; `dd.run`
  accepts an `SDE` object (built on the fly). Point-process models in config files go through a model file
  (documented).
- **Sage-preparsed literals in the simulators (bug 11).** `SDE.simulate`, `noise_samplers.simulate_ou_paths`,
  `simulations.delay_sde_sim.simulate` and the Meixner sampler coerce `Integer` / `RealNumber` arguments
  (`default_rng(Integer)` / `round(RealNumber)` raised TypeError in a Sage notebook).
- **Mean-field solvers (bug 3).** A residual that overflows to inf / NaN under `np.errstate(all='ignore')` is
  penalized (1e10) instead of passed on: NaN compares False against the acceptance tolerance and could pass as a
  root. After `fsolve`, the legacy and heterogeneous iteration solvers check the residual at the returned point
  (`SADDLE_RESIDUAL_RTOL = 1e-8` relative) and raise instead of returning a non-converged saddle (the engine's
  saddle check caught such points later, with a less specific message).
- **Invalid `DAEDALUS_PHASE_J_ENGINE` (bug 9)** raises at the `compute_cumulants` entry (before: at the first
  Phase J subset, after the enumeration).
- **Implicit point-process saddle equation (bug 13).** An SDE model with a point process whose mean-field equation
  of a continuous field references that field (a conductance term `(Es - v)*Conv(g, n)`) was emitted as an explicit
  `vstar = ...` substitution and failed the saddle check; it now uses the residual `.equation` route (the saddle
  equals the exact root to 1e-13; simulation agrees within 2 SE).
- **`symbolic_loop` (bug 4)** refuses a placement-wrapped contribution (`phase_j_placements`) by name
  (`kernel_edge`) instead of an AttributeError (not reachable through `symbolic_cumulant`, which refuses
  conductance vertices first).
- Checked, already fixed on main: the delta-in-heaviside DC gain (bug 1, #20; end-to-end test added) and the
  stale `legacy_baseline[multipopulation_test]` slow test (bug 6, refrozen in #20; passes on both engines).

#### Added: kernel direct terms at vertex legs; the bare-delay refusal is per line (item 56)

Numbers move only for models with a bare hard delay at a vertex leg (`x*x(t - d)`, a bare delay inside a rate
nonlinearity), which now run the (n, n) Padé approximant instead of item 47's strictly proper `[n-1/n]` rerun: the
two approximants differ at finite n and converge to the same limit (below). No other model is touched (a strictly
proper kernel takes exactly the old code paths).

- **Direct terms.** A conductance kernel whose frequency image is `c + (strictly proper)` (`g = c δ + Θ Σ C e^{λt}`:
  the all-pass (n, n) image of a bare delay) is folded into the dressed line it sits on, `c G + g_smooth ∗ G`, on
  both Phase J routes: the pole route's dressed mode sum gains `c C_α` at every propagator rate (and the δ part
  `c D` on a line with one), the state-space route's series connection gains `C = [d₂ c₁, c₂]`
  (`realize_kernel(..., direct=True)` keeps `d`). A kernel edge (a leg time: a line that cannot be dressed, e.g. a
  kernel rate equal to a pole of the line) holds only a strictly proper kernel and refuses a direct term by name
  (`kernel_direct_term`; Phase J status `failed`); the delay driver then still reruns with `[n-1/n]`. Counters
  `conv_lines_dressed_direct`, `conv_kernel_direct_refused`.
- **Run-level route.** `run_route`, `diagram_route` and the memory pre-pass (`diagram_estimate`) count such a kernel
  as a dressed line (`_ModelOps.kernel_route`). Before, they asked the kernel-edge realization, so a bare-delay
  model was predicted "not covered" and every diagram was charged pole tuples: a delayed negative-feedback model
  at k = 1, l = 3 (Padé 4) was estimated at ~3,300 GB of pole tuples and refused before any pass ran (item 47's
  retry never fired); it is now covered, estimated at 13 GB of state-space states.
- **Refusal per line (D10).** `bare_delays_on_delta_lines`: a bare delay is refused only when it delays a field whose
  propagator has a δ part (a point-process field; read with every delay factor at its Padé image, both parities)
  into a vertex with a second δ-carrying physical leg (a spike train inside a nonlinearity, `exp(w n(t - d))`,
  `n(t) n(t - d)`). Phase J refuses the same case by name, per diagram (a direct term on a δ line at a vertex where
  another line keeps a δ part). Now run (item 47 refused any bare delay in a model with point-process lines): a bare
  delay of a continuous field inside `exp` in a spiking model; a delayed spike train times a continuous field
  (`n(t - d) (1 + eps v)`); a bare delay of a spike train in a linear term.
- **Checks.** `x*x(t - d)`, (n, n) at n = 4 and 8: equal to the same rational kernel declared without a delay (state-space
  route 3e-16; pole route 1.9e-10 at one loop, its usual precision) and to the Markov embedding of the (n, n) kernel
  (n auxiliary linear fields from a balanced realization, no kernel code): n = 4 tree 1.3e-15, one loop 7e-15;
  n = 8 tree 8e-16, one loop 1.1e-14. (n, n) and `[n-1/n]` at n = 8: same limit (loops within 1e-4 of max|C|, as n = 4 vs 8). A bare
  delay of the voltage inside `exp` equals the same process with the delay moved into the voltage equation (one loop
  1.3e-15). A delayed spike train times the voltage, Padé (2, 2), vs its Markov embedding: tree and one loop
  <= 1.4e-12. A linear Hawkes neuron with a bare delay of its own spike train (direct term on a δ line): k = 3 tree vs
  its Markov embedding 4e-14; its two-point one loop is exactly 0.
- **Convergence.** On a delayed negative-feedback model with the delay 5x the decay time, the l = 1 mean correction
  at (n, n) is -5.1727e-4 / -5.17376e-4 / -5.17376e-4 at n = 4 / 8 / 16, while `[n-1/n]` gives -3.21e-4 / -4.20e-4 /
  -4.71e-4: the strictly proper approximant converges slowly for loop terms. With `order='auto'`, `x*x(t - d)` stops
  at n = 8 in 9.5 s; item 47's double pass needed `[15/16]` (113 s).

#### Fixed: tier-1 residues no longer clipped across entries; memory budgets follow the machine (item 53)

- **Residue clip.** The tier-1 (exact-membership, double-precision) residues were zeroed below 1e-12 of the LARGEST
  entry at their pole. Membership is exact, so every member residue is exactly nonzero and the clip could only drop a
  genuine one: two fields whose units differ by 1e14 lost their cross propagator. On `x' = -x + 1e-14 z`,
  `z' = -z/2` the pole engine returned `C_xz = 0`; it now matches the Lyapunov covariance to ~1e-16 (the default
  state-space engine was already right). The certificate (`_residue_error`) always checked the unclipped values. The
  numerical cofactor tiers keep their clip (no membership there: the clip removes the noise of non-member entries).
  **No number moves on the zoo:** none of its 35 entries has a tier-1 residue below 1e-12 of its pole's largest
  (smallest ratio 1.6e-8), so the stored residues are identical.
- **Default memory budget.** With `PHASE_J_MEM_BUDGET_GB` / `SPATIAL_MEM_BUDGET_GB` unset the budget is
  `min(6 GB, 0.6 x RAM - what this process already holds)`, at least `max(0.5 GB, 10 % of RAM)`
  (`final_integral.default_mem_budget`; the process is measured by its macOS `phys_footprint`, or resident + swap on
  Linux). Same 6 GB on a machine with >= ~13 GB of RAM and a small process; 3.3 GB on an 8 GB machine with 1.5 GB
  held; 6 GB on a 64 GB machine even with 40 GB held. `compute_cumulants` reads it once per run (the pre-pass and the
  running guard use the same value). Explicit values are unchanged.
  Every refusal says how its budget was set; the Phase J pre-pass refusal also states the machine's RAM and the
  estimate; `result['phase_j_memory']['budget_rule']` records the rule.
- **Periodic lattice sum.** `periodic_inverse_ft` in d=3 built the whole `(2 n_cut + 1)^3` lattice (~9 arrays,
  13.6 GB at L = 30, k_max = 60: the peak of `tests/test_spatial_correlator.py`, which no guard saw). The sum
  factorises over the phase axis, so it now runs one n1 slab at a time (0.3 GB peak for the same case). Values move
  by the order of the additions only: d = 1 bit-identical, d = 2 <= 4e-16 relative, d = 3 <= 8.5e-15 relative where
  the sum cancels (old and new are equally far, <= 5e-15, from an exact `math.fsum` of the same terms).

#### Fixed: mean-field saddle (round 3, item 52)

- **Kernels whose integral is not 1.** `Conv(K, constant)` at the saddle is now `K(omega = 0)·constant`
  (the kernel's integral, read off its frequency image; a delay factor counts 1). It was the constant
  itself (a normalised-kernel rule), so a model with `freq_image='2/(1+I*omega)'` (or a negative or
  per-entry integral, an indexed kernel, a delayed and scaled kernel) stopped with "mean-field saddle
  does not satisfy its own equations" although its mean-field equations carry the integral. A kernel
  symbol left in a hand-written mean-field text (`w*g*nstar`) is also taken at its integral by the
  check and the iteration solver (it was 1). A model now equals its twin with the normalised kernel
  and the integral moved into the weight (saddle, tree, one loop). A kernel whose image is singular at
  omega = 0 (no finite integral) is refused by name. Normalised kernels: the expansion is unchanged
  (no `1*` factor is introduced). Expand-cache bundles written before this change are re-expanded for
  models with a non-unit kernel only.
- **The selected root is checked.** The numerical mean-field check inside the expansion runs at the
  model's default parameters and root 0 (and not at all on an expand-cache hit); the root chosen by
  `fixed_point_index` at the run's parameters is now checked too, so an equation that shares only
  the default root with the action stops instead of computing at a saddle the action does not have.
- **Negative roots of a field with no declared domain.** Such a field is seeded on `[0, 5s]` only (the
  historical box; widening it moved other models' roots), so `fixed_point_index` could not reach the
  root -1 of `x - x^3`. The selectable roots are unchanged; the same seeds are re-run on the mirrored
  half-line and a warning names every root found there with the remedy (`domain='real'` on the
  field, or `mf_dae_seed_box`); the roots are in `mf_unsearched_roots` of the DAE result.
- **`analyze` parameter point** = `compute_cumulants`' (declared defaults completed by the question's
  values, mean-field defaults never merged; same error wording for a partial dict).
- **Scalar models** (no `.population()`): a `set_mf_equation` text may name a saddle bare
  (`'a*exp(Em + c*vstar)'`, not only `vstar[i]`); it was a parse error. Results equal the 1-element
  population form bit for bit.

#### Added: matrix-exponential output for every k >= 3 chamber, and the exact delayed tree's pairing and spectrum (item 54)

No computed number moves (output side only).

- **k >= 3 middle chambers.** `slice_me` (and the new `Cumulant.me_slice(j)`) now gives triples on every chamber of a
  slice, including a swept leg between two other legs, where two segments move (one grows, one shrinks).
  `SubsetSS.me_chain` returns the moving segments and the fixed maps between them; `MEChainBlock` holds them as one
  Kronecker triple (the last segment transposed, a middle one as `I ⊗ G`). Values are evaluated segment by segment
  and equal `total_C_by_ell` / `dd.run`'s slices to <= 4e-16 (oscillator + red noise, k = 3, l <= 1; the double-well
  OU at mu = -1, k = 3, l <= 1). Mode lists of a middle chamber combine the two segments' own decompositions; they
  reproduce the values to <= 1.2e-14. A middle chamber is a finite interval: it has values and modes, no Laplace
  transform.
- **Exact delayed tree (k = 2).** `Cumulant.me` now holds the exact delayed tree of `api.delay` itself
  (`ExactDelayTreeSide`): values, Laplace transforms (the history segment on `[0, d]` from the boundary-value
  stack plus the exact tail through `Δ(z) = zI − A − A_d e^{−zd}`) and the spectrum `H(iω) N H(iω)ᴴ − D0 N D0ᵀ`
  with the true delayed image. `integrate_against` and `spectrum` with `method='auto'` are now exact for delay
  models with the exact tree (before: refused). Scalar delayed OU: spectrum vs the closed form 5e-16; Laplace
  transforms (powers 0-2) vs an independent spectral quadrature <= 6e-14. Explicit `method='quad'` to infinite
  lag stays refused. The exact tree has no triple and no finite mode list (`me.mode_reasons`; `Cumulant.modes`
  leaves it out with a warning).
- **Mode merging.** More than 2,000 mode terms are now merged by exact rate first and clustered row by row
  (before: an n x n comparison, 7 GB for the middle chambers). Smaller lists take the old path unchanged.

#### Added: point-process front end (round 2, item 46)

No existing number moves (no existing model uses the new paths).
- `SDE.point_process('n', rate, population=..., size=...)` declares a spike train with conditional
  intensity `rate`: the action gains `nt*n - (exp(size*nt) - 1)*rate`, the mean field `n* = size*rate*`
  (kernels at omega = 0). Rates may use population sums, declared functions and `Conv(K, x)` of any field,
  the spike train itself included (spike-only Hawkes); models made only of point processes build. The
  spike-only exponential Hawkes declared this way equals the hand-written builder model bit for bit
  (action, saddle, tree and one loop). Models with a point process emit their saddle equations as
  `set_mf_equation` (the route hand-written point-process models use) when every equation reads
  `x = rhs` at Dt = 0.
- A product of declared kernels inside `Conv` (`Conv(h[i]*g[i,j], n[j])`) is their convolution: a kernel
  `h_g` is declared with the product of the frequency images and the sum of the delays.
  `Kernel.alpha(tau)` (`t/tau^2 exp(-t/tau)`); a `Kernel` product with one non-trivial factor keeps that
  factor's image text unchanged.
- Bare parameters indexed by one population are indexed automatically in SDE texts (`a` -> `a[i]`).
- `api/pp_simulate.py`, `SDE.simulate(...)`: a time-discretised simulator of point processes, drifts and
  Gaussian sources, kernels realized exactly as state-space filters (delays by a ring buffer);
  `cross_covariance` and `mean_rate` estimators with batch-means errors.

#### Fixed: package warnings were hidden under `sage -python` and the Sage kernel (item 51)

`DelayOrderWarning`, `PhaseJIncompleteWarning`, `PhaseJEngineFallbackWarning`, `PhaseJDivergenceWarning`,
`PhaseJNquadFallbackWarning`, `GroupedDeltaRecordsDroppedWarning` and `StaleResultWarning` were never shown in Sage.
Results still carried their `status` and cap fields. No number moves.

- **Cause.** The Sage launcher sets `PYTHONWARNINGS=ignore`, and the helper `_ensure_warning_shown` is meant to
  override only that environment ignore. Sage also installs message-specific numpy ignores at startup
  (`numpy.ndarray size changed`, ...). The helper took the first filter that applies to the category, saw such a
  targeted filter, took it for a user's decision, and did nothing.
- **Fix.** The helper now takes the warning's `message` and module (or `stacklevel`), and skips a filter whose message
  or module pattern does not match. A category-wide `ignore`, a matching message or module filter, `error`, and a
  line-targeted filter are still respected. The filter it adds is for that one message, so a user's ignore aimed at
  another message of the same category keeps working. Applies to both copies (`api.save`, `engine`'s
  `final_integral`) and every call site.

#### Added: delay kernels inside nonlinear terms (round 2, item 47)

A hard delay inside a nonlinear term (`x*x(t - d)`) used to return `status='incomplete'` with a warning. It now runs.
No number that was computed before moves: only runs that came back incomplete take the new path.

- **What was refused, and why.** A delay kernel at a vertex leg is a conductance kernel on the Padé-approximated model.
  - A delayed strictly proper kernel was already routed like any other kernel (dressed lines / kernel edges, exact
    tree). Examples: a delayed leg operator `Kernel.delay(d) * membrane * alpha` inside a rate nonlinearity, or
    `x*Conv(g, x)` with a delayed filter.
  - A bare hard delay has the all-pass (n, n) image `(-1)^n + (strictly proper)`. Neither kernel route takes that
    direct term (`kernel_not_strictly_proper`), so Phase J skipped those diagrams.
- **Strictly proper retry.** When a run comes back incomplete and the model has a bare delay kernel
  (`bare_delay_kernels`), `compute_delayed` reruns the orders with the strictly proper `[n-1/n]` Padé approximant of
  every delay factor (`pade_model(..., proper=True)`, `pade_mn_coefficients`).
  - That approximant also converges to `exp(-s d)`, and its denominator is Hurwitz. The tree stays exact.
  - The record says `approximant='pade_proper'`, and `G3_DELAY_APPROXIMATED` and the ledger name `Pade (n-1, n)`.
  - The first attempt's warnings are dropped when the retry completes, and re-emitted otherwise.
- **Refused by name.** A bare hard delay at a vertex leg in a model whose propagator has a δ part (point-process
  fields) is refused, for example `exp(w n(t - d))` with `n` a spike train. The nonlinear term multiplies δ
  functions, so no approximant is run. The result is `incomplete` with `cap` starting `refused`, plus the
  `DelayOrderWarning`. The remedy is to give the delay a filter.
- **Validation** (`tests/test_delay_nonlinear.py`; `test_delay_kernels.py`'s never-silent test now expects the
  completed run):
  - **Spike-only exponential Hawkes with a delayed alpha × membrane leg operator inside `exp`, against its voltage
    form** (the delay in the linear voltage equation), Padé 4.
    - 1 neuron: tree to 3e-16, one loop to 3.5e-16.
    - 2 neurons: tree to 2.8e-15 on 49 lags.
  - **`x' = -a x + b - c x x(t - d) + noise`.**
    - The `[n-1/n]` run equals the same rational kernel declared without a delay (tree 7e-18, one loop 3.8e-16).
    - The loops at n = 4 and 8 agree to 1.1e-5 of max |C|.
    - Exact tree + one loop is within 0.6 SE of the Euler–Maruyama simulation (Richardson in dt) at every lag. The
      tree alone is 5–7 SE off.
- **Limit.** The 2-neuron spike-only one loop exceeds the Phase J memory guard (estimate about 122 GB over 1,446
  diagrams: every leg is a dressed line). Its voltage form runs.

#### Added: noise library, finite-moment Levy sources and samplers (round 2, item 48)

No existing number moves (existing noise tests unchanged, zoo bit-identical).

- **New sources** (`api/noise.py`, each one per-unit-time CGF, parameter domains checked with clear errors):
  `VarianceGamma`, `NormalInverseGaussian`, `CGMY` (`Y < 2`, `Y` not 0 or 1), `TemperedStableSubordinator`,
  `Meixner`, `BilateralGamma`, `Skellam`, `NegativeBinomialProcess`, and `SumSource` (`a + b` on sources).
- **New jump laws** for `CompoundPoisson`: `Geometric`, `PoissonJump`, `NegativeBinomial`, `Logarithmic`,
  `DoubleExponential` (Kou), `AsymmetricLaplace`, `Hyperexponential`, `PhaseType(alpha, T)` (rational MGF;
  Erlang / hyperexponential / Coxian), `ChiSquared` (exact gamma), and the moment-series laws `LogNormal`, `Weibull`
  whose CGF is a **truncated series** to a requested `order`: cumulants or action orders above it are refused.
- **Composites**: `merton_jump_diffusion`, `kou_jump_diffusion`, `levy_driven_ou`. Telegraph noise and Markov-modulated
  Poisson are expressible at the action level (state-dependent Poisson) but not usable by the perturbative pipeline;
  see `docs/noise_library.md`.
- **Samplers** (`api/noise_samplers.py`): `source.sample(dt, size, rng, values=None)` and `jump.sample(rng, size)`;
  exact increments for every source except CGMY with `1 < Y < 2` (controlled Gaussian small-jump approximation) and
  `Meixner` (numerical inversion of the exact density); `simulate_ou_paths` is the Levy-driven OU simulator.
- Inline `sde:` config blocks accept the new source types.

#### Fixed: a declared parameter named `e` was read as Euler's number (round 2, item 48)

`SDE().parameter('e', ...)` with `Poisson(rate='e', size='e')` printed `exp(1)` in the action while the drift's `e` was
the parameter, a silently wrong action and saddle. `SDE.parameter / field / define_function / define_kernel / noise(expose=)`
now refuse the names `e`, `pi`, `I`, the keywords (`sum`, `for`, `in`, `if`, `else`, `range`, `Dt`, `Conv`, `t`, `theta`),
Python keywords and the elementary-function names with `SDEError` (population labels only refuse `Dt`, `Conv`, `t`,
`theta` and Python keywords: they are loop-set labels, never symbols; `I` stays a valid population name).

#### Added: matrix-exponential triples for conductance-kernel and delay-kernel models (S4b)

Models with field × Conv (conductance) vertices now have triples, so `integrate_against` and `spectrum` are exact
for them as well (one resolvent solve per block instead of quadrature). No number of `C_tau`, `C_tau_by_ell`, the
callables or the δ records moves: the only change on the compute path is one attribute on a closure. Main and the
branch are bit-identical on the tracked zoo, including its alpha-kernel conductance (Hawkes) model.

- **Why they had none.** A conductance diagram whose kernel slots can be filled in several ways is integrated as
  the average of its placements. That averaged contribution did not expose its subsets. It now lists its placements
  and their weights (`phase_j_placements`). Each placement's dressed lines and kernel edges give ordinary blocks, and
  the triples are built from them; `slice_me` (k ≥ 3) does the same.
- **Large blocks.** A block beyond the dense forms (`MAX_BLOCK` = 2000 states; the 2-neuron delayed-alpha Hawkes one
  loop has 2,772) is paired by forward substitution over the segment's downsets. Each step is one shifted
  Kronecker-sum solve, and no dense matrix is formed. It equals the dense route to 5e-15 where both apply.
- **Per loop order.** `method='auto'` now splits by loop order. Orders with triples are exact. The others fall back
  on quadrature, with an `MEFallbackWarning` that names them and their reasons. Each fallback is counted in
  `Cumulant.me_fallbacks`, and the pairing's `method` is then `'exact+quad'`. Before, one order without triples
  sent every order to quadrature.
- **Delay kernels.** With the exact delayed tree (the default for k = 2), the tree is not one matrix exponential
  (method of steps), so it has no triple; `me.reasons[0]` says so. Before, `me` silently gave the Padé tree there,
  which differs from `C_tau`'s exact tree. Pairing or transforming that tree by quadrature to infinite lag is
  refused up front with `MEUnavailable`, because its method-of-steps stack is capped. The loop orders alone
  (`ell=1`), `delay={'tree': 'pade'}` or a finite `tau_max` (spectrum) work. With the Padé tree, every order has
  triples.
- **Validation** (`tests/test_s4b_conductance_triples.py`):
  - **Two-rate conductance kernel (tree + one loop).**
    - Triples equal `C_tau_by_ell` on 201 points: tree 0, loop 4.2e-16.
    - Pairing equals the quadrature of the Phase J callables to 5.5e-16 (7 ms instead of 11 s).
    - Spectrum equals QAWF to 8.5e-13 (2 ms instead of 37 s for three ω).
  - **Alpha kernel at equal rates (exact double pole) against its Markov embedding.**
    - Triples equal the embedding's `C_tau` to 1.3e-15.
    - Pairing and spectrum equal the embedding's to 4e-18.
  - **Delayed exponential conductance kernel and the delayed-alpha 2-neuron Hawkes network (Padé 4).**
    - Triples equal `C_tau_by_ell` (Hawkes: tree 1.3e-16, loop 2.4e-15; 262,472 loop states).
    - Pairing and spectrum equal an independent dense resolvent to 1.2e-13 and 1.7e-14 (2 s and 5 s).
    - The Phase J callables' quadrature of this model refuses long lags (a stiff segment), so there was no
      quadrature value to compare with.

#### Fixed: float-literal coupled drifts, partial `parameters=`, NaN lags (round 2, item 49)

Three input-handling bugs; no number that was computed before moves (the zoo is bit-identical wherever it built).

- **A coupled linear drift written with float (or integer) literals failed the mean-field self-check.**
  `SDE('m').field('x').field('y').drift('x', '-1.0*x + 0.8*y').drift('y', '-0.3*x - 0.6*y')` raised
  `AssertionError: mean-field saddle does not satisfy its own equations` at `expand()`; the same model with named
  parameters built.  A model with no declared parameters got no numerical fallback in the check
  (`_default_fundamental_point` returned `None` for the empty point), and the symbolic substitution of a coupled
  implicit saddle (`x* = 0.8 y*`, `y* = -0.5 x*`) is simultaneous, so it cannot verify it alone.  The empty point is now
  a point; a wrong mean-field equation in a parameter-free model is still rejected (test).  The float and named forms
  now give bit-identical actions, saddles and tree + one-loop cumulants.
- **A partial `parameters=` dict raised a bare `NameError` in the mean-field DAE.**  Parameters the caller omits now
  come from the model's declared `default=` (the given values win); one with no default raises a `ValueError` naming it
  in `compute_cumulants`.  The solvers below it only fill (a no-default parameter that the saddle equations never read,
  such as a noise strength, must not stop a direct solve), and the DAE residual names an unbound parameter instead of
  the bare `NameError`.  A `mean_field=True` parameter's default is never merged in (it would pin the saddle in the
  legacy iteration solver; `compute_cumulants` did this with `parameters=None` already, now with no explicit dict
  either).  The caller's dict is not modified.
- **A NaN lag was evaluated as tau = 0.**  The Ito equal-time nudge made the comparison against NaN pass, so a NaN
  in `tau_grid` returned the equal-time value without a word; the same test in the k = 2 `total_C` callable (and so
  `dd.cumulant(...).total_on_grid`) and in the k >= 3 base-lag pinning did the same.  `compute_cumulants` now raises
  `ValueError` up front for a NaN / infinite entry of `tau_grid`, `spatial_grid` or `spatial_points` and for a NaN /
  infinite `tau_step` / `tau_max` (both Phase J engines); the callables and the base lags pass a NaN on to the engine,
  which refuses it.  `dd.Config` rejects non-finite `tau_grid`, `chi_grid`, `times`, `tau_max`, `tau_step` and
  `kpoint_base_lags`.

#### Changed: balanced truncation of the state-space realizations; the delay order controller measures its cost (item 45)

Every per-entry realization of the propagator and every conductance kernel (`api/_realization.py`) is now
**balanced-truncated** to a relative H-infinity tolerance (default `1e-12`, 1 % of the pipeline's default accuracy
target): states are discarded only while
`2 * sum(discarded Hankel singular values) <= tol * sigma_1` (`sigma_1 <= ||G_pr||_inf`), so each entry stays within
`tol` of the exact one.  Exact Kalman minimality (S1) removes only exactly uncontrollable / unobservable states; the
Padé blocks of a delay inside a feedback loop leave many that are numerically negligible, and those are where the
state-space route's cost goes.  Model-agnostic: any rational kernel (Padé delays, Erlang chains, alpha kernels,
`freq_image`) and any closed loop.

- **Where numbers move.**  Only where a state is discarded, and then within the tolerance.  A realization with nothing
  to discard is returned as it was (the same object), so minimal, well-conditioned models keep every number bit for bit.
  Zoo A/B (35 entries, caches off, each in its own process): 29 bit-identical, 5 within 1e-19 .. 1e-16 with no entry
  truncated (main's own cross-process scatter), walls equal.  The one public model that moves is
  `multipopulation_test` (k = 2, tree): 64 of its entries hold states below the tolerance (per-entry states
  1024 -> 664 in total), C(tau_0) 2.1780065123783054e-4 -> 2.178006512350135e-4, largest change 2.8e-15
  (1.8e-12 of max |C|), wall 3.6 -> 3.3 s.  Erlang-16 OU (one loop): 17 -> 13 states at 1e-10, total moved by 1.8e-12
  (bound 1.2e-11), 1.8 -> 1.0 s.  Unchanged (bit-identical): colour-noise OU at its double pole, an alpha-kernel
  conductance and its Markov embedding, Erlang-8 OU, Padé-16 / Padé-24 OU.
- **Method.**  Square-root balanced truncation on directly computed Gramian factors (Hammarling on the complex Schur
  form, then a real factor from `[Re Z, Im Z]`): an eigendecomposition of a computed Gramian loses the Hankel singular
  values below ~1e-8 of the largest, and on the graded Padé-16 entries of a two-neuron loop (`|b| ~ 1e12`) the
  truncation then missed its tolerance by 1e-7.  Every truncation is checked a posteriori (`|G - G_r|` on a frequency
  grid against `tol * sigma_1`), retried with one more state, and refused after three retries; a non-Hurwitz result is
  never returned.  The global `A, B, C` (Hurwitz test, exact delay tree, stationary covariance) are not truncated.
- **Control.**  `DAEDALUS_REALIZATION_TRUNCATION=<tol>` (`0` / `off` / `exact`: none, the realizations of before),
  `api._realization.truncation_scope(tol)` (process-wide, nested scopes restore; clears the state-space route's
  realization cache when the tolerance changes), `realize_K(..., truncation=)` / `realize_kernel(..., truncation=)`.
  `Realization.truncation` records the tolerance and, per entry, the states before / after, the Hankel singular
  values, the bound, the sampled error.
- **Delays** (`api/delay.py`).  Every Padé order runs with its own truncation (`delay={'truncation': 'auto'}`,
  default): the first at `min(tolerance in effect, delay tol / 100)`, every later one at 1 % of the relative loop
  error predicted for it (from the exact-vs-Padé tree times the loops' share of max |C|, and from the doubling
  estimate, both carried by the loops' n^-3 decay; capped at 1e-4).  The predictions overestimate the loops' Padé
  error ~10x on the benchmarks, so the per-entry bound is ~10 % of it and the observable moves by far less
  (`delay={'truncation': x}` pins one tolerance, `0` exact).  Delayed-alpha two-neuron demo (k = 2 one loop, 49 lags):
  Padé-8 took 653 s with exact realizations and 109 s at the automatic 1.1e-6 (mean per-entry states 22 -> 12.5),
  the loop within 1.5e-8 of max |C| of the exact one (Padé error estimate 1.6e-4).  The automatic order now runs
  n = 4 (72 s) -> 8 (107 s, predicted 113 s) -> 16 (1172 s at 2.4e-8, 20 states per entry instead of 38; predicted
  822 s), estimate 1.7e-5 at n_max; before, it stopped at n = 4 (n = 8 predicted at 16x).  Padé-16 with exact
  realizations is not measured (the proxy puts it at 3-4 h).
- **Order controller.**  The next order's wall is the measured wall of the last order times the ratio of a cost proxy,
  mean per-entry states^2 x (1 + mean Taylor stiffness `min_p ||(a - mu)^p||^(1/p)`), of the two orders' truncated
  realizations (the next one is built from the true kernel matrix and truncated as that run will be, a few seconds),
  times the proxy's error on the step before.  On the demo the walls follow states^2.2 x stiffness^0.95 within 10 %
  over five runs (71 .. 654 s); the old `(n_next / n)^4` floor predicted 16x where the run took 10.6x and stopped the
  demo at n = 4.  The old rule stays as the fallback: no realization, a last run not entirely on the state-space
  route, or a proxy that does not grow (a delay only inside a nonlinear term).  `r['delay']` adds `stop_reason`,
  `cost_by_order`, `predicted_walls_s`, `cost_correction`, `truncation_tol`, `truncation_by_order`,
  `loop_error_pred_by_order`.
- **`api/symbolic_loop.py`.**  The reference run of `symbolic_cumulant` / `symbolic_triple` uses exact realizations
  (its checks are at 1e-12); a `result=` computed with truncated entries is refused by name (`truncated_reference`).
- **Tests.**  `tests/test_kernel_model_reduction.py` (22 tests, ~15 s).  The default suite passed with the tolerance
  at 1e-10 except one test that compares an Erlang-32 OU tree pointwise to 1e-11 (a lag where |C| is 0.003 moved by
  5e-12); at the 1e-12 default it passes -- one reason for the default.

#### Added: symbolic loop corrections as rational functions (C3)

`dd.symbolic_cumulant(model, k, max_ell, external_fields=...)` (`api/symbolic_loop.py`) returns the **equal-time
cumulant** of every loop order as an exact rational function of the model parameters (and, on request, of the
saddle); with `kernel=` (a `Kernel`, k = 2) the pairing `∫ L(τ) C(τ) dτ` of exponential and polynomial × exponential
kernels; `dd.symbolic_triple` gives the matrix-exponential triples `(G, x0, c)` of S4 over the same field, with the exact
rational spectrum and, for small triples, `C(τ)` itself through C2's `tree_modes` (OU quartic one loop:
`−3εD²/μ³ (1 + μ|τ|) e^{−μ|τ|}`).  Nothing else changes: no existing output moves.

- **Method.** The state-space route's stationary sweep, replayed exactly over `QQ(params)`: `G(s) = K(s)^{-1}` is
  inverted symbolically and each entry realized by its companion form (no roots, no `RootOf`, no distinct-pole
  assumption: double poles are ordinary); every resolvent of a Kronecker-sum block is one fraction-free (Bareiss)
  solve over `QQ[params]`, factored once per block.  The diagrams, prefactors, Wick permutations, δ edges and subset
  structure are the pipeline's own (a reference run at numeric parameters), and the result is checked against the
  pipeline there (`check()`: relative 1e-12, raises otherwise).
- **Validated.** OU quartic: one and two loops equal the exact Boltzmann O(ε), O(ε²) coefficients symbolically
  (`−3εD²/μ³`, `24ε²D³/μ⁵`); colour-noise OU at the exact double pole: `−12εD²(1 + 2μτ_c)/(μ³(1 + μτ_c)³)`; a
  three-field k = 3 one loop with five symbolic parameters (7 s) equals the pipeline at two parameter points to
  1e-12; pairings equal S4's exact `integrate_against`; the triples equal `C_tau`.
- **Symbols and saddle.** `symbols=` chooses the symbolic parameters (others enter as the simplest rational of their
  double); `saddle='auto'` (default: the saddle stays a number when a second mean-field solve shows it does not move
  with the symbolic parameters, else it becomes a variable), `'numeric'` (exact for the reference saddle; warns when
  evaluated elsewhere if the saddle moves), `'symbol'` or `{name: closed form}` (checked against the reference).
  Exports: Sage, sympy, LaTeX, JSON, a numpy callable; `dd.` and `api.symbolic_out` re-export the entry points
  (the latter through a module `__getattr__`).
- **Limits.** Refused by name (`SymbolicLoopRefused.reason`): Kronecker blocks over `max_block` states (default 64;
  e.g. an Erlang-2 kernel at one loop needs 81 and takes ~2 min with five symbols), conductance kernels
  (`kernel_edge`), non-local noise kernels, subsets off the state-space route, spatial models, parameters entering
  through square roots (e.g. correlated-noise amplitudes `√(D1 D2)`: keep them numeric).

#### Added: switch-on transients (non-stationary stage 1)

`compute_cumulants(..., t_min=T, times=...)` (and `dd.Config(t_min=..., times=...)`) computes the cumulants of a
**switch-on transient**: the fluctuations about the selected mean-field saddle start at zero at the absolute time
`T` and relax towards the stationary state.  Every vertex-time integral then has the lower bound `T`, and the
cumulant is a function of the absolute external times.  Without `t_min` nothing changes: stationary runs are
bit-identical to main on the tracked zoo (11 entries, k = 1, 2, 3, ℓ ≤ 2).

- **Engine.** The state-space Phase J route starts its sweep at `T` from the empty state; its first segment, `T` to
  the first external time, is a finite matrix exponential in place of the stationary resolvents.  Only this route is
  used: `statespace.switch_on_scope(T)` (a thread-local bound read by the route; `final_integral.PHASE_J_T_MIN` is not
  touched, so a stationary evaluation in another thread is unaffected) turns every subset the route declines, and
  every call-time refusal that would fall back on the pole engine, into `SwitchOnUnsupported` (a `ValueError`).
  `DAEDALUS_PHASE_J_ENGINE=poles`, a noise-kernel box (a kernel the model builder could not Markov-embed), a
  pole-cluster deformation the route cannot bypass, delay kernels (`api.delay`), spatial models, the grouped
  integrator, δ components, `tau_grid` and `output_npz`/`output_csv` are refused with a message, and so are
  `save_npz`/`save_csv`, `dd.cumulant`, `me_output.from_result`/`slice_me`, `dd.plot_cumulant`,
  `generate_report` and `tree_covariance_from_result` of a switch-on result.
- **Times.** `times` is `(n, k)` (absolute times of the k legs) or 1-D (every leg at `t`: the equal-time cumulant;
  `dd.Config` also takes an `(lo, hi, n)` tuple).  Every time must be `>= t_min`.  Equal times are exact ties on the
  state-space route (no Itô nudge; the stationary `C_tau` keeps its τ = 0⁻ sample).  The result holds `times`,
  `C_times`, `C_times_by_ell`, `t_min` and `nonstationary` (the record of the question); `C_tau`, `C_tau_by_ell`
  and `tau_grid` are `None`; `total_C` / `total_C_by_ell` and the closures kept in `phase_j_by_ell` take absolute
  times.
- **Convention for embedded noise.** A noise kernel that the model builder Markov-embedded (an auxiliary field, e.g.
  an exponential CGF kernel) starts at zero too, so the noise is switched on with the system
  (`result['nonstationary']['embedded_noise_states']`; the same numbers as writing the embedding by hand).
- **Initial-state term (tree level).** `initial_covariance=C0` (k = 2) adds `R(t1 - T) C0 R(t2 - T)^T`, with `R`
  the response from the model's realization: `C0` enters like a noise vertex at `T`, so this is the whole k = 2 tree
  term of a Gaussian initial state.  `max_ell >= 1` with it raises (loops need an initial-state vertex), and the
  model's response at `0+` must be the identity (a first-order model whose response index r is the equation of
  physical field r).  A deterministic initial offset of the mean is not offered: its tree is the full nonlinear
  relaxation (a time-dependent saddle, stage 2).
- **Analyzer.** The ledger's `stationary_state` line is replaced by `switch_on_state`; G6 sizes its accuracy estimate
  on `C_times`.
- **Validation** (`tests/test_switch_on_transients.py`, inline models): OU tree vs the closed-form time-dependent
  Lyapunov solution, ≤ 1e-13 (measured ~3e-16); OU + εx³ one loop, quadratic drift (k = 1 mean, k = 2, k = 3 tree and
  one loop), colored OU (Markov embedded, all leg pairs) and a non-reciprocal two-field model vs the exact multi-time
  moment hierarchy of the Itô generator (`tests/_ito_moments.py` with frozen coordinates for the earlier times),
  ≤ 1e-10 (measured ≤ 2e-15); the initial-covariance term vs the closed form; `t_min = -80` vs the stationary run
  ≤ 1e-13; t1 = t2 = t_min gives exactly 0; thread safety (callables from 8 threads next to a stationary
  evaluation).

#### Added: delay kernels (Padé loops with an automatic order, exact tree, kernel-aware stability, delay simulator)

A kernel image with a factor `exp(-I*omega*d)` (`d > 0`) is a delay kernel. Before, the analyzer stopped such a
model (`G3_NONRATIONAL_KERNEL`); now `compute_cumulants` computes it (`api/delay.py`). No number of a model
without a delay changes.

- **Front end.** `define_kernel('g', delay='d')` (the hard delay `delta(t - d)`; with `freq_image=` / `time_expr=`
  a delayed filter), `define_kernel('g', kernel=Kernel.delay('d') * Kernel.exponential('tau'))`
  (`api.delay.Kernel`: `delay`, `exponential`, `gamma`, `image`, products), a `time_expr` shifted by a delay
  (`dirac_delta(t - d)`, `heaviside(t - d)*exp(-(t - d)/tau)/tau`), a hand-written `exp(-I*omega*d)` image, and in
  the SDE front end a delayed field `x(t - d)` in a drift (rewritten as `Conv(delay_d, x)`). Mean field: the
  delay leaves `g_hat(0)` unchanged (exact).
- **Loops: Padé (n, n).** Each delay factor is replaced by its diagonal Padé approximant `P_n(-z)/P_n(z)`,
  `z = i omega d` (all-pass, exact rational coefficients), and the approximated model runs on the ordinary
  pipeline (state-space route). `delay={'order': 'auto'}` (default) doubles `n` from 4 until the estimated
  relative error of the observable, `max_tau |C_n - C_2n| / max_tau |C_2n|` (conservative), meets `tol`
  (default `1e-8`), or `n_max` (16), or the predicted wall time of the next order exceeds `wall_max` (900 s); an
  order the engine cannot finish, or whose difference grows, ends the search at the previous order.
  An order whose Padé tree is more than 100 % of max|C| from the exact tree is skipped (a long delay needs a higher
  order). A hard delay inside a nonlinear term (`x*x(t - d)`) is not computed: its all-pass image has a direct term,
  which the conductance-kernel routes need to be zero; the run is `incomplete`, the record says so and a
  `DelayOrderWarning` names the remedy (a delayed filter, `Kernel.delay(d) * Kernel.exponential(tau)`).
  `output_npz` / `output_csv` are written from the final result. `delay={'order': n}` fixes the order. The result carries `result['delay']` (orders run, walls, the estimate, whether
  the target was met), the finding `G3_DELAY_APPROXIMATED` (a warning when the target is missed, with a
  `DelayOrderWarning`) and the ledger entry `approximation delay:<kernel>`.
- **Tree: exact.** For `k = 2` the tree is the exact stationary covariance of the linear delayed system
  (`api/_delay_exact.py`: `K = K0 + K1 e^{-s d}` realized exactly, a boundary-value problem on `[0, d]` plus the
  method of steps as one matrix exponential; delayed noise included, and the smooth part of the response of a
  field with an instantaneous (delta) response; the `delta(tau)` part itself stays the pipeline's delta record), not
  its Padé approximation. `delay={'tree': 'pade'}` or `DAEDALUS_DELAY_EXACT_TREE=0` switches it off. One delay
  value; several delays, a boundary-value problem whose error estimate exceeds 1e-8 (a delay long against the
  decay rates: `d max|Re eig A|` above about 17) or a method-of-steps stack above 1,500 states (short delays on long
  tau grids) use the Padé tree (recorded in
  `result['delay']['tree_note']`).
- **Stability: G2-K.** The saddle's stability is decided from `det K(s)` with the TRUE kernel images
  (`api/_kernel_stability.py`): Lambert W for one field with one delay, otherwise the argument principle on a
  half disc whose radius is certified against the leading term. An unstable delayed saddle is refused by name
  (`G2_UNSTABLE_SADDLE`, with the roots), a marginal one by `G2_MARGINAL_MODE`, a neutral system by the new
  `G2_NEUTRAL_DELAY`, an advance (`exp(+i omega d)`) by `G3_ACAUSAL_POLE`. G2-K also runs for every model with a
  rational kernel (the mean-field linearization replaces each kernel by its integral): `G2_KERNEL_STABILITY`
  (info) when stable; `G2_KERNEL_UNSTABLE` (a warning; an error only under the strict policy) when not; a
  propagator pole below the axis that G3 used to call `G3_ACAUSAL_POLE` is then named `G2_UNSTABLE_SADDLE` (both
  errors; a `policy.downgrade` that accepted such a model must now name `G2_UNSTABLE_SADDLE`).
  New traits: `delays`, `kernel_stability`, `n_unstable_roots`; `propagator_class` gains `'delay'`
  (`'delay-rational'` in a result).
- **Simulator.** `simulations/delay_sde_sim.py`: Euler-Maruyama with a ring-buffer history for delayed SDEs
  (numpy, vectorised over paths, no processes), an autocovariance estimator with batch-means errors, and the exact
  stationary autocovariance of the scheme itself for linear scalar models (to separate statistics from the
  `O(dt)` bias).

#### Fixed: the state-space route returned wrong numbers for some high-order rational kernels

`api._realization.realize_K` now checks a strongly graded realization a posteriori (balancing range above 1e6, or a
Lyapunov margin <= 0 although the exact Routh test says Hurwitz): the rounded matrices' stationary output
covariance must be positive semidefinite and agree with the frequency-domain integral of the realization's own
transfer function. A realization that fails is refused (`RealizationRefused('ill_conditioned')`), and the
state-space Phase J route falls back to the pole route; the run's census counts it
(`ss_declined_by_reason['realization_ill_conditioned']`) and the analyzer adds the warning `G6_SS_ILL_CONDITIONED`.
Seen on a 4-field model with a Padé-24 / Padé-32 kernel on its cycle (|B| ~ 1e24 .. 1e35): C came out ~1e13 with
`G6_ACCURACY` reporting no problem; it now equals the pole engine's value. Ordinary models (balancing range within
[0.125, 8]) are not checked and unchanged; the scalar Padé-24/32 realizations pass the check and stay on the route.

#### Added: cumulants as matrix-exponential triples (C0, state-space stage S4)

`dd.cumulant(result)` of a k = 2 result computed on the state-space route now also holds its smooth part as
matrix-exponential triples, one per loop order and side: `C(τ) = c e^{G|τ|} x0` (`api/me_output.py`). Each block is
one subset of one diagram in one Wick permutation: the last segment of the state-space sweep (`G`, block lower
triangular over the downsets of the subset's poset), the state that enters it (`x0`) and the contraction of the last
external time (`c`), with the same prefactors and Wick compensation as `C_tau`. No number of `C_tau`, `C_tau_by_ell`, the callables or
the δ records moves (only attributes are added to the Phase J closures; main vs branch bit-identical on the tracked
zoo); `spectrum()` values move within its old quadrature tolerance (its default is now the exact route).

- **`c.me`** (`CumulantME`): `c.me(tau)` at any τ (exact τ = 0⁻ by default; `C_tau` samples τ = −1e-6), the dense
  `c.me.sides[ell]['+'].triple()`, and `c.me_reasons` when a loop order has no triples (a subset on the pole route,
  a pole-cluster deformation run, grouped Phase J, a finite `PHASE_J_T_MIN` or Phase J flags changed since the run,
  call-time fallbacks during the run); `c.me` is `None` for a loaded file or k ≠ 2.
- **`c.modes`**: `{ell: [ModeTerm(coeff, rate, power, side)]}` derived on request from the eigenvalue clusters of the
  final generators (taken in the per-edge triangular basis, where the eigenvalue sums are exact): `τ^n` terms for
  repeated eigenvalues; `MEConditioningWarning` when clusters nearly coincide or the modes cancel (the triples are
  unaffected); loop orders without triples are left out with an `MEFallbackWarning` (`None` when none has them).
- **`c.integrate_against(kernel, contact=...)`**: `∫ L(τ) C(τ) dτ` for `Kernel` terms (exponential, polynomial ×
  exponential, two-sided) by one triangular resolvent solve per block, `q! c (zI − G)^{−q−1} x0`, and for a
  `RationalKernel(num, den, side)` (rational Laplace transform) by one Sylvester solve per block; the δ components
  with item 23's contact rule (a kernel that jumps at 0 needs `contact='exclude' | 'positive' | 'negative' | 'mid'`).
  `method='quad'` integrates the Phase J callables instead.
- **`c.spectrum(omega)`**: now `c (iω − G_+)^{−1} x0 + c (−iω − G_−)^{−1} x0` + the δ floor by default (`method='auto'`):
  spike reset one loop 46 s -> 7 ms per ω, linear Hawkes 2.8 s -> 0.1 ms per ω, values equal to the quadrature
  within its tolerance. `method='quad'` (Feature B's QAWF) and `'grid'` remain; without triples `'auto'` falls back on
  the quadrature with an `MEFallbackWarning`.
- **k >= 3**: `api.me_output.slice_me(result, j)` gives the triples of `dd.run`'s slice `j` on the chambers where
  the swept leg is first or last (one moving segment); a chamber where it lies between two legs is refused by name
  (`several_lags`).
- **Validation**: tree level equal to item 23's symbolic modes at numbers within 1e-13 (OU, underdamped OU, the
  cubic-denominator chain, an Erlang-8 kernel), mode lists included; loop level equal to `C_tau_by_ell` within 1e-13
  on 201 points (OU quartic to two loops, colour noise at the exact double pole, an oscillator driven by red noise;
  measured also: spike reset one loop 8.6e-15); pairings equal to quadrature within 1e-10; the linear-Hawkes spectrum
  equal to its closed form within 1e-12; k = 3 slices equal to `dd.run`'s (`tests/test_c0_output.py`).

#### Changed: `inverse_transform` / `pair_with_kernel` run on the stage-C2 engine

`symbolic_out.inverse_transform` and `pair_with_kernel` (and the `TreeCovariance` methods) now use item 23's engine
by default: any denominator degree (roots of factors of degree >= 3 are `RootOf` symbols), the half-plane split
stated as assumptions (`'assumptions'`, `'roots'`, `'modes'` in the `inverse_transform` dict) instead of sampled
around `reference`. Behaviour that changes: a factor of degree >= 3 no longer raises `ClosedFormUnavailable` (nor
falls back on quadrature in `pair_with_kernel`); a split that flips near the reference is stated as a condition
instead of refused; a reference point where poles coincide is moved to a nearby generic point (refused when the move
would change a sign condition); the default `pair_with_kernel` applies `simplify_full` to its closed form (as the
pre-C2 route did); the lag symbol is `tau_lag` when `tau`
is a parameter of the entry (it was `tau`, the parameter itself). The pre-C2 routes
remain behind `inverse_transform(..., legacy=True)` and `pair_with_kernel(..., any_degree=False)`.
The `legacy=True` dict has no `'modes'`, `'roots'` or `'assumptions'` keys. `dd.` now exports `tree_covariance`,
`tree_covariance_from_result`, `TreeCovariance`, `tree_modes`, `ModeSum`, `Kernel`, `KernelIntegral`,
`integrate_against`, `inverse_transform`, `pair_with_kernel`, `AssumptionViolation`, `ModeTerm`, `CumulantME`,
`RationalKernel`, `MEUnavailable`, `MEFallbackWarning`, `MEConditioningWarning` and `slice_me`.

#### Fixed: compiled kernels accept `c*dirac_delta(t)` terms (`api/model_compiler.py`: `causal_exponential_ft`)

A kernel given by `time_expr='(0.3*dirac_delta(t) + exp(-t/tau)/tau)*heaviside(t)'` was refused by the compiled
transform ("factor dirac_delta(t) is not t^n or an exponential"). A `c*dirac_delta(t)` term, with or without a
`heaviside(t)` factor, now contributes the constant `c` to the frequency image: the delta of a causal kernel keeps
its full weight, as in the mean-field integral of the SDE front end, and a factor depending on `t` is evaluated at
the delta (`exp(-t/tau)*dirac_delta(t)` gives 1, `t*dirac_delta(t)` gives 0). The image has the form of the
hand-written `c + 1/(1 + I*omega*tau)`, and the k = 2 tree and one-loop cumulants of a model using it are bit-identical
to those of the same kernel given by `freq_image` (builder and SDE front end alike). A `dirac_delta(t - d)` with
d != 0 is a delay, not a local term: it is refused with a message pointing to `Kernel.delay`. Kernels without delta
terms are unchanged (same image string). No number of an existing model moves.

#### Documentation: the state-space route and the engine choice

`ARCHITECTURE.md` describes `api/_realization.py` and `engine/integration/time_domain/statespace.py` (what each
does, how the engine is chosen, `DAEDALUS_PHASE_J_ENGINE=poles`, `PHASE_J_MEM_BUDGET_GB`); the README has a short
section on the engine choice.

#### Changed: the state-space route is the default Phase J engine (`DAEDALUS_PHASE_J_ENGINE=poles` switches back)

The state-space subset integrator below is now used for every subset it can take; `DAEDALUS_PHASE_J_ENGINE=poles`
restores the pole engine everywhere (the dispatch then never imports the module, and its numbers are those of
0.2.0 + the items above, bit for bit). `default` and `statespace` name the new default.

- **Coverage.** Plain propagator rows (S2) and now also conductance kernels: a kernel edge uses the kernel's own
  realization (`api._realization.realize_kernel`: the exact chain of its frequency image, or of the Laplace
  transform of its time image), a dressed line the kernel cascaded with its propagator entry. A kernel whose rates
  are equal or within 10 % of each other, which the pole engine refuses by name, is taken by the route
  (alpha-kernel conductance in kernel form at τ2 = τ, 0.999 τ, 0.9 τ, 0.62 τ: within 5.5e-16 of its Markov
  embedding at one loop). Noise-kernel boxes, spatial models, symbolic prefactors and the legacy Θ(0) mode stay
  on the pole routes (`ss_declined_by_reason` counts them).
- **Fallback, never silent.** A routed subset whose evaluation is refused at call time (a stiff segment beyond the
  triangular route's size, a segment system over the memory budget or too large, a non-finite value, a
  `THETA0_CONST_ROW_MODE='legacy_clip'` set after the build) is evaluated by the pole route for that call; its
  modesum plan is built only then. Counted by reason (`ss_fallback_*`), reported in `result['phase_j_engine']`
  and as the G6 warning `G6_SS_FALLBACK`, and warned once per run (`PhaseJEngineFallbackWarning`).
  `DAEDALUS_PHASE_J_SS_FALLBACK=0` raises instead. Strict policy semantics are unchanged (G6 is recorded, never
  raised).
- **Engine census.** `result['phase_j_engine']` (also `result['route']['phase_j_engine']`): the run's prediction,
  subsets on the route and on the pole routes (by reason), fallbacks, and what happened to the pole-cluster
  deformation; the hook payload has `engine` and `ss_fallback`; the analyzer adds `G6_PHASE_J_ENGINE` and an
  assumptions-ledger entry `phase_j_engine`.
- **Pole-cluster deformation** is not run when the route covers the run (it uses no poles): one Phase J pass
  instead of one per node (7 on an exact double pole), `pole_clusters['status'] == 'bypassed'`. If a subset
  declines the route after all, the run is repeated with the deformation.
- **Memory pre-pass** (item 35): a diagram the route takes is charged the route's own bound (stationary states x
  the τ batch, plus one segment system) instead of its pole tuples. Runs the pre-pass refused although the route
  finishes them now run (an OU with an Erlang-4 kernel at two loops: pole estimate 2.15 GB, route bound 12 MB (+ one segment system, checked when built),
  0.75 MB held).
- **τ batching** inside `total_C_batch`, `eval_per_diagram_batch` and the grouped `total_C_batch` (serial paths).
- **Grouped Phase J**: a group the route covers is integrated per diagram on it (`handled_by='statespace_perdiag'`;
  the grouped evaluator shares pole-tuple work, which the route does not have).
- **Analyzer**: G3's repeated / close / unresolved-cluster findings are informational when the route covers the
  run; G6's accuracy estimate includes the route's a-posteriori estimate (refinement residuals and the Taylor
  tolerance per step on the final contraction's magnitude), so it is no longer 0 or missing on routed runs.

- **Walls** (zoo, warm, default vs `poles`, timed alone): small models within 1.05x (0.9-1.03x), except the
  spike-reset family with hundreds of tiny delta subsets (1.08-1.15x: per-subset overhead of the route);
  `single_population_quad_exp_test` 11.3 s -> 1.1 s, `multipopulation_test` 3.0 s -> 1.2 s (no deformation runs),
  `ou_quartic` k = 4 two loops 6.0 s -> 4.0 s, grouped spike reset 1.8 s -> 1.0 s; a 3-field oscillator with red
  noise, k = 3 one loop, 5.6 s -> 2.4 s. Runs the pole pre-pass refuses now finish at the default budget:
  `quadratic_hawkes_alpha` in 9.2 s and `dendritic_quad_soma_sigmoid` in 24 s (pole estimates 185 / 113 GB; a
  25-evaluation 50-digit sample within 6.9e-15 / 7.8e-16), and an alpha-kernel embedding at two loops under
  0.5 GB (pole estimate 20 GB) in 10 s.
- **Accuracy** (default engine, 80-digit oracles): Erlang-n kernels n <= 32 within 5.1e-15 (pole engine up to
  3.2e-13), all-pass n <= 24 within 1.1e-14 (pole engine up to 6.5e-14), their one loops within 2.5e-15 (pole
  engine up to 6.1e-13); a two-population linear Hawkes model with Erlang-16 kernels 5.3e-15 (2.1e-14).

Movers (default vs `poles`, the model zoo in process; relative to max |C_ℓ| over the grid; every other entry
within 1e-13). In each case the value that moves is the pole engine's, checked against an independent reference:

| model | ℓ | max |Δ| / max |C_ℓ| | reference | state space vs reference | pole engine vs reference |
|---|---|---|---|---|---|
| `ou_quartic_two_dim_color_corr` (k = 2, x–y) | 1 | 3.4e-13 | 50-digit sweep of every subset on the exact realization | 1.8e-16 | 3.4e-13 |

(A local zoo model whose one-loop cumulant is exactly 0 reads 0 on the route and ~1e-21 of rounding on the pole
engine.)

#### Changed: faster mean-field-sector check in `FieldTheory.expand`; no number moves

`_verify_and_zero_mf_sector` ran `simplify_full` on every coefficient of the mean-field sector and, for the numerical
tier, solved the saddle again for each of them. A coefficient that interval arithmetic (128 bits, at a point that
satisfies every assumption) shows to be a nonzero function cannot simplify to 0, so `simplify_full` is now skipped
for it (it still runs to word a failure, so the error text is the same); float literals are read as the exact
rationals they are and the proof is kept only if it survives perturbing the floats by 1 +- 1e-8 (all up, all down,
alternating), so an expression that cancels to float dust, e.g. floats an ulp apart, is not claimed nonzero. The saddle is solved once per check. The expansion, the warnings and the
errors are identical on all 75 zoo items that expand (dict order, `is_trivially_equal`; in-process A/B against the
previous source). Whole zoo `expand()` 15.9 s -> 6.4 s; public models, taylor order in brackets, seconds before -> after:
`ou_quartic_two_dim_color_corr` (2) 0.78 -> 0.36, `dendritic_quad_soma_sigmoid` (4) 0.48 -> 0.16, `coupled_rd_2species_1d`
(4) 0.45 -> 0.20, `multipopulation_test` (4) 0.28 -> 0.18; a polynomial-coefficient model with float coefficients
(order 4) 12.9 -> 7.4 (the check 6.9 -> 1.4).

#### Fixed: front-end and engine leftovers

- A scalar model (no `.population()`) whose closure is given by `set_mf_equation('nstar', 'a*exp(Em + c*vstar[i])')`
  (or `define_function`) failed the mean-field self-check: its saddle parameters carry no `indexed_by`, so the
  heterogeneous mean-field solver skipped them and the residual check bound them to 1. They now belong to the model's one
  (auto-injected) population, as in the 1-element population twin; saddle, action and cumulants equal the twin's
  bit for bit. Mean-field results of all 75 zoo items unchanged. One visible difference for a scalar model that
  already built on main with `set_mf_equation` only: `mf_values` of the result now holds the saddle (it was empty);
  every cumulant is bit-identical.
- `SDE._kernel_dc` of `(c*dirac_delta(t) + h(t))*heaviside(t)` returned `c*heaviside(t) + h~(0)`. The delta of a causal
  kernel counts with its full weight (`c + h~(0)`, as for `c*dirac_delta(t) + h(t)*heaviside(t)`); a `t`-dependent factor
  of the delta is evaluated at 0.
- A numpy overflow in the mean-field DAE residual raised `KeyError('__import__')` once warnings are shown (the
  expression is evaluated with empty `__builtins__`); the evaluation now runs under `np.errstate(all='ignore')` (values
  unchanged: `1/(1 + exp(300*x^2))` saturates to a valid number).
- A population named `pop` whose members are not `range(n_pop)` (`n_pop` = number of populations) had `pop` bound to
  `range(n_pop)`: `sum(... for i in pop)` silently covered the first member only (size 3 beside nothing else), or ran
  past the last one (size 2 beside two more populations). `pop` now iterates over the population's own members, and
  `SDE.population('pop', size=N)` is accepted. Unchanged when they agree (every existing model).
- `SDE` with `interpretation('stratonovich')` and an lhs whose `Dt` coefficient is not constant (`(1 + x^2)*Dt*x`) was
  read as `a = 1` and got the correction of `Dt*x`; it is refused ("cannot read a constant Dt coefficient"). A `Dt` of
  another field in the equation (`Dt*x + Dt*y`) is still read off as before.
- Checked and not changed: scalar-mode `linear_stability` on `exp(field)` works on main (pinned by a test); actions with
  `D[0](f)(...)` atoms cannot be written in the front end and the legacy `taylor()` leaves such an atom unexpanded when
  its argument is not a bare field (the per-atom route already agrees with it where it expands).

#### Tests

- `tests/test_phase_j_legacy_baseline.py::test_heavier_entries_reproduce[multipopulation_test-k2-l1-default]` (slow)
  failed at 1.5e-4: its stored tree level and `polytope_m0_direct` predate the pole-cluster resolution (item 21). The
  original baseline file is untouched; the two moved arrays and the counter are refrozen beside it.

#### Added: state-space subset integrator (opt-in, DAEDALUS_PHASE_J_ENGINE=statespace) (`engine/integration/time_domain/statespace.py`, one dispatch point in `final_integral.py`; no number moves with the flag unset)

A Phase J subset integral evaluated without poles, residues or pole tuples, on the realization layer
`api/_realization.py` (`G(t) = C e^{At} B + D δ(t)`). Each smooth edge carries the minimal realization of ITS
propagator entry, so its tensor axis has that entry's state count (1 for a driven filter that does not feed
back), never the model's. The integral is a sweep over the downsets of the subset's causal poset: before the
first external time the states are resolvents of Kronecker sums (per-edge Schur basis, vectorised back
substitution, refinement on the residual), computed once per subset; between external times one truncated-Taylor
action (Al-Mohy & Higham 2011, trace shifted, no squaring) of a block-triangular generator, or, for a stiff
segment (fast rates over a long gap), the exponential of the same generator written upper triangular in the
per-edge Schur basis (reset-diagonal Padé, exact decay on the diagonal); at an external time the edges that end
or start there are contracted or opened. Every vertex order of the poset is integrated at
once (the block exponential's Dyson series is the sum over linear extensions), and equal or near-equal poles
need no special case. The module docstring has the method and a proof sketch.

- **Opt in** with `DAEDALUS_PHASE_J_ENGINE=statespace` (read when a diagram is integrated; `poles`/`default`/unset:
  off). A subset is taken only when every row is a plain propagator edge of a model the realization layer
  accepts with a Hurwitz `A`; noise-kernel boxes, conductance kernel edges and dressed lines and spatial models
  (including the spatial certification's per-momentum calls) stay on the pole routes, and so do grouped Phase J's
  grouped integrals (its per-diagram fallback goes through `integrate_diagram` and may use the route). A stiff
  segment beyond 600 states that would need more than 2,000 Taylor steps raises rather than lose accuracy. The
  route is recorded in the per-subset hook (`path='ss'`), in
  `subset_diagnostics` (`evaluator='statespace'`) and in `ss_*` counters (`ss_declined_by_reason` says why a
  subset was not taken). δ-subset enumeration, exact δ elimination and Feature B records are unchanged; the
  integrated δ weights use the new closure.
- **Θ(0) and ties**: constant rows follow Phase J's Itô rule and tie order (`_const_row_verdict`); equal
  external times form one event. A finite `final_integral.PHASE_J_T_MIN` is a first finite segment.
- **Batching**: `SubsetSS.evaluate_many` and the `tau_grid_prefetch(points)` context evaluate a whole τ grid per
  subset (stationary state shared, the last segment stepped); not yet wired into `total_C_batch`.
- **Memory**: the cached stationary states are charged against `PHASE_J_MEM_BUDGET_GB` (item 35's running
  total) before they are allocated; a segment system whose working set alone is over budget is refused.
- **Hook**: `path='ss'`, `plan=None`, `integrand` = the closure the pole routes would integrate, and the new
  key `ss_subset`.
- **Accuracy** (50-digit oracles: Lyapunov plus the exact first-order Hartree shift, or a 50-digit sweep on the
  exact realization): OU with cubic drift 2e-16; OU driven by a colour field at gaps 0.1 / 1e-4 / 1e-8 / 0 (an
  exact double pole) <= 6e-16 where Phase J has 1.1e-10 / 4e-13 / 4e-13 / 4e-13; Erlang-8 / Erlang-16 kernel
  3.1e-15 / 5.2e-15; Padé-8 / 16 / 24 kernel 5.5e-15 / 9.2e-15 / 3.7e-14; alpha-kernel conductance through its
  Markov embedding 3.5e-16, also at equal kernel rates, where the kernel form is refused. Stiff segments (rates
  1e2..1e5 over gaps up to 50, a colour field at τc = 1e-4) <= 3.4e-15. Model zoo, every subset the route
  takes: within 1e-13 relative (or 1e-15 of the residue scale) of Phase J, and where not, the 50-digit sweep
  sides with the state-space value in every case (Phase J off by up to 1.8e-13 of the scale on m = 2 polygons
  with heavy cancellation).
- **Speed**: 3-field oscillator with red noise, k = 3 one loop at tau = 0, `dd.run`: 1.6 s end to end (value
  0.16673618, unchanged); 4-field oscillator with two filters: 28 s at 0.95 GB peak footprint (Phase J with zero-mode
  pruning: about 18 min per point). OU with an Erlang-4 kernel, two loops: 1.5 s at 0.7 GB (Phase J: 576 s at
  3.2 GB, values equal to 1.2e-14); its tree + one loop on a 201-point grid: 0.77 s batched, 1.45 s per point
  (Phase J 16 s).

#### Fixed: front-end small bugs: SDE `time_expr` kernels, `Conv` on a bare field, `_is_single_function_call` (`api/sde.py`, `api/model_compiler.py`; no number moves)

- **SDE `time_expr` kernel integral.** `SDE._kernel_dc` (the kernel's integral for the mean-field equations) asked
  Maxima for the integral and failed on a difference of exponentials,
  `(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)` ("requested additional constraints"). It now falls back
  to the closed form `n!/a^(n+1)` of `causal_exponential_ft` at omega = 0, with the rates positive by declaration
  (`domain='positive'`, indexed parameters included); no Maxima assumption is made, so `assumptions()` is
  unchanged. A kernel the closed form cannot read (rates not declared positive, no `heaviside`) raises an
  `SDEError` naming the cause. Kernels Maxima already integrated take the same path as before.
- **Scalar models, `Conv(r, n)` on a bare field** raised "no canonical coercion from `_FullPhysicalField`"; `Conv`
  now unwraps the size-1 field wrapper as the declared functions do. The model equals its one-element-population
  twin bit for bit.
- **`_is_single_function_call('f(a) + g(b)')`** returned True (it checked only that the parentheses balance, not
  that the first one closes at the end); also `f(a)*g(b)`, `f(a)(b)`. No model in `models/` has an equation text
  on which the old and new answers differ.
- `SDE.notes` is rebuilt on every derivation (it already was; a test now pins that `equations()`, `build()` and
  `show()` do not duplicate entries).

#### Changed: Phase J leaves exact-zero residue modes out of its pole-tuple plans; k >= 3 one loop from tens of minutes (or out of memory) to seconds (`final_integral.py`, `grouped_integral.py`, `daedalus.py`; no number moves)

Every propagator edge carried one mode per pole, including poles whose residue in that entry is exactly 0. A
driven filter that does not feed back (red noise on an oscillator) makes `K` block triangular, and the entries
below it miss its poles. The modesum plan is the Cartesian product of the edges' modes, `n_poles^E` tuples of
~1.1 KB each, and the zero tuples were only discarded when the sums were folded. At k = 3 one loop (E = 9 edges)
this was 394k tuples instead of 4.4k for a 3-field oscillator with one red-noise filter, and 100.5 M (~108 GB)
instead of 0.97 M for a 4-field one with two.

- **Pruning** (`PRUNE_ZERO_MODES`, env `DAEDALUS_PRUNE_ZERO_MODES=0` turns it off): every plan, per diagram and
  grouped, enumerates only the tuples whose residues are all nonzero. The test is `== 0` on the stored residue,
  with no tolerance. The propagator stores an exact 0 only where an entry has no denominator, where a pole is
  outside the entry's exact membership, or below its 1e-12 noise clip; those terms were exactly 0 before.
  The pruned plan is bit-identical to the unpruned one: the groups keep the order of their first occurrence in
  the full product (replayed on numpy, never materialised), and the decay margin and the batched-kernel
  thresholds use the unpruned data. One exception: on the legacy (non-DD) routes, a zero-coefficient tuple can
  no longer trigger an overflow bail to quadrature.
- **Zero-coefficient groups** (`DD_BATCH_SKIP_ZERO_GROUPS`): the batched poset kernels compute chains only for
  groups whose folded coefficient is nonzero.
- **k >= 3 `dd.run` slices**: the total is the sum of the per-loop-order slices, in `total_C`'s order. Before,
  every diagram was evaluated a second time, which halves the evaluations.
- Results: 3-field oscillator, k = 3 one loop at tau = 0, `dd.run`: 2,224 s -> 4.9 s (0.47 s warm). The value is
  unchanged: 0.1667361824735574, equal to the pre-change value. 4-field oscillator with two filters, same run:
  out of memory (~108 GB of plans) -> 1,946 s at 1.83 GB peak footprint (machine shared with other runs). Zoo
  in-process A/B against main, warm-up discarded: 40 of 42 entries bit-identical, the other 2 raise the same
  error on both sides, total wall 0.99x.

#### Added: Phase J memory budget `PHASE_J_MEM_BUDGET_GB` (default 6) (`final_integral.py`, `api/compute.py`, `api/_grouped_phase_j.py`)

Phase J refuses rather than exhausting memory. A pre-pass at the start of Phase J estimates the tuples every
plan will hold (`estimate_modesum_plans`: per diagram, the nonzero modes of each propagator edge, the δ-subset
sum; diagrams whose prefactor is exactly 0 are left out when the count without that test is over budget)
times 1,100 bytes; it costs milliseconds (about 8 µs per diagram). A running guard before each plan is allocated
keeps the exact total of the run (`phase_j_memory_scope`). The estimate is an upper bound: δ-subsets that
build no plan are counted. On a spike-only two-neuron Hawkes model it is 9.3 M tuples where 1.8 M are built.
So the pre-pass refuses outright only above 8x the budget (`PHASE_J_PREPASS_REFUSE_FACTOR`; the unpruned
108 GB case above); between 1x and 8x the running guard decides at each plan. Either raises
`PhaseJResourceError` (`dd.PhaseJResourceError`, a `MemoryError`) under every analyzer policy, before the
offending plan is allocated. The message gives the numbers, the largest diagram or subset, and how to raise the
budget (`PHASE_J_MEM_BUDGET_GB` in GB, or `off`). The estimate is in `result['phase_j_memory']`, in the analyzer report
(`diagnostics['phase_j_memory']`) and in the assumptions ledger (`phase_j_memory_budget`). No zoo entry is near
the default budget.

#### Changed: pure-integrator mean-field equations are accepted (`api/model.py`, `api/_mean_field_dae.py`)

`.equation(lhs='Dt*h[i]', rhs='-om*T[i]')`, whose LHS vanishes at Dt = 0, used to be refused with a request to
move the algebraic part to the LHS. It now builds: the saddle text is derived as for the moved form
`lhs='Dt*h[i] - (-om*T[i])', rhs='0'`, and results are bit-identical to that form. The equation itself is kept
as written for the joint DAE solve. Three cases are refused with a clear message:

- an RHS without any field (a random walk: nothing pins it), at build time;
- a constraint that is not linear in any field with a constant coefficient (use `set_mf_equation`), at build time;
- a system whose equilibrated Jacobian is singular at the root (the integrated field does not feed back
  anywhere), in the saddle solve.

Models without a pure-integrator equation are untouched. Tests: `tests/test_zero_mode_pruning.py`,
`tests/test_mf_lhs_coefficient.py`.

#### Added: state-space realization of propagators (internal, not yet used by Phase J) (`api/_realization.py`)

`realize(prop, num_params)` writes the propagator `G(s) = K(s)^{-1}` (s = iω) as `C (sI − A)^{-1} B + D`,
`G(t) = Θ(t) C e^{At} B + D δ(t)`, without an eigenvalue or a numerical root. The algebra is exact over QQ at
the parameter point (each double taken at its exact binary value) and is rounded once at the end:

- Kernel denominators are factored exactly, so a kernel's own rates come back even from an expanded image.
  Each rate becomes a unit-gain chain block (Erlang kernels are the Erlang chain). An irreducible factor becomes
  a tridiagonal Schwarz block built from its Routh table. If a Routh pivot vanishes, an exact integer shift is
  tried; the companion matrix is used only as a last resort.
- A singular `K1` (algebraic fields, δ parts) is reduced by an exact index-1 Schur complement. `compare_D`
  checks `D` exactly against the pipeline's `D_delta`.
- An exact Kalman reduction gives a minimal realization, globally and per entry of `G`.
- A power-of-two balancing follows, then an exact Routh–Hurwitz report and a Lyapunov margin.

Models without a realization are refused by name: non-rational kernel, descriptor of index > 1, polynomial
part of degree ≥ 2, unbound or spatial symbol, singular pencil. `impulse(t)` exponentiates `A` without a dense Schur step: a
triangular `A` gets a triangular Padé that resets the diagonal and superdiagonal exactly after each squaring,
and any other `A` gets scipy's `expm`. `stationary_covariance(Q)` uses a Lyapunov solve with residual
refinement and refuses when `A` is not Hurwitz. Nothing in the pipeline
calls the module yet; no number changes. Tests: `tests/test_realization.py`.

#### Fixed: high-degree rational kernels and propagators gave C = 0 or a wrong C without a warning (`api/_propagator.py`, `api/analyze.py`)

The exact propagator tier decided that a pole `p_k` of `Q(ω)` (the common denominator) is not a pole of entry
`(i, j)` by an absolute test, `|Q_ij(p_k)| > 1e-9`, on the entry's monic denominator. At high degree its rounding
alone exceeds that, so every pole was skipped and the residues were zero. A scalar OU with an Erlang-n kernel,
`dx = (−x − 2 (g∗x)) dt + sqrt(0.2) dW`, `g(s) = (1 + s/n)^−n`, had tree `C(0)` = 0.1354441967 exact and
0.0000000000 computed at n = 8. n = 16 and 32, all-pass kernels `B_n(−s)/B_n(s)` of degree 8 to 24 (`B_n` the reverse
Bessel polynomial) and two-population Hawkes models with Erlang-8/16 kernels also gave 0; the degree-4 all-pass
kernel was 4 % off. Low degrees (n ≤ 4 here) were right.

- **Membership is exact.** The entries' denominators split `Q` into pairwise coprime factors (gcd over `QQ[i]`), so
  which poles belong to which entry is exact. When every entry has `Q` itself as denominator, nothing is decided
  numerically. Otherwise each computed root is matched to its factor by Newton distance, and every factor must
  receive exactly its degree in roots. A match that is not clear-cut takes the exact route below.
- **The double-precision residues are certified.** Poles are Newton-refined on the exact `Q` and residues
  recomputed at 30 digits. If they differ by more than 1e-11 relative, the propagator takes the exact 50-digit route
  of `api._pole_clusters` (roots of each exact factor). Monomial-basis roots of a degree-17 or higher denominator
  lose 9 or more digits in double precision (Erlang-16: C 2.6e-11 off; Erlang-32: 10 % off).
- **Never silent.** `prop['pole_analysis']` records `membership` (`'shared'` or `'labelled'`) and `residue_route`
  (`'double'`, `'exact'` or `'failed'`; absent when the pole-cluster path computed the residues). The analyzer reports
  `G3_RESIDUES_HIGH_PRECISION` (info) or `G3_RESIDUES_NUMERICAL_TIER` (warning: the exact route failed and the
  numerical tier was used), and has the trait `residue_route`. That last case also issues an
  `api._propagator.PoleResidueWarning`.
- **The numerical tier** (used when the exact tier fails) had sibling defects, fixed the same way. Leading
  coefficients of `num(det K)` below 1e-12 of the largest were trimmed as noise, but at high degree they are
  genuine: Erlang-16 kept 4 of 17 poles and was 25 % off. Roots of the untrimmed polynomial where `det K` vanishes
  are now restored. Refined poles were rejected by an absolute `|det| > 1e-6`; they are now also accepted when
  `det K` vanishes relative to a small circle around them (the polynomial-fit fallback, used when symbolic root
  finding fails, uses that relative test alone). An entry of the `D_delta` probe below 1e-12 is zeroed only when it
  is also noise relative to the matrix. A vanishing `det'` and the fixed-degree polynomial-fit fallback now warn
  instead of giving silent zeros.

Every zoo model is bit-identical (in-process A/B, warm caches), and the overhead is below 1 % of the wall time.
OU trees with Erlang-n (n ≤ 32) and all-pass (degree ≤ 24) kernels, and the one-loop εx³ term at n = 8, now match
exact state-space references at 80 digits; `tests/test_residue_membership.py` checks them to 1e-11 (measured:
≤ 3.3e-13). Two-population linear Hawkes models with Erlang-8 and Erlang-16 kernels match too (measured ≤ 2.2e-14;
not in the test suite, about 2 minutes each).

#### Fixed: vertices with a field times a convolution, `x·Conv(g, y)`, were wrong, dropped or took days (`final_integral.py`, `api/_expand_cache.py`, `api/_diagrams.py`, `engine/core/vertices.py`)

Such a vertex (a "conductance" synapse `v·(g∗n)`, any `x·(g∗y)` coupling) puts its `y` leg at its own time `s`,
joined to the vertex time `t` by the kernel `g(t − s)`. Three defects; which one a run met depended on the cache:

- **Warm cache: diagrams dropped.** The expand cache did not restore the kernel attachments that
  `extract_vertex_types` needs, so every run after the first built a plain vertex with the kernel placeholder left in
  its coefficient; Phase J failed those diagrams ("stripped integrand contains unexpected free symbols {z_g_...}")
  and left them out of `C_tau` without a word. On a one-population spiking model with a `v·(g∗n)` term, 21 of the
  29 one-loop k = 2 diagrams were missing. The bundle now stores the attachments; an older bundle is re-expanded only
  when a kernel symbol sits in an interaction coefficient (models whose `Conv` is linear keep their cache). The
  typed-diagram cache could not store these vertices at all (a closure in the vertex, and `VertexType.__getstate__`
  saved only the subclass's own slots); both fixed, and the typed-diagram stage name gets a `_convleg` suffix only
  for a model with such a vertex.
- **Wrong routing.** The kernel delay was applied to every edge at the vertex whose physical leg had the attached
  leg's population (and at both ends of an edge), which in a one-population model is every edge: the kernel
  collapsed to an instantaneous coupling. Now only the line whose head is the vertex at the attached leg (field
  and population) carries the kernel.
- **Days of quadrature.** The delay `τ` was the integration variable, so every edge into the attached leg had a
  three-term constraint row, which the exact box-free routes do not take: the subsets went to nested scipy
  quadrature (m up to 6), days per τ point (a zoo run was killed after 5.5 days). Now a kernel that is a finite
  exponential sum (single exponential, difference of exponentials, ...) is folded into the propagator of its line,
  `(g ∗ G)`, again a pole sum without δ part: no extra time variable, no extra row, no `TAU_KERNEL_CAP` box (that
  box truncated the kernel, 4.5e-5 for τ_g = 5). The propagator's rounding-level residues at the kernel's own
  rates (a pole of the kernel's image is a zero of `G`) are dropped as exact zeros; they multiplied the
  pole-tuple count for nothing. Any other kernel (alpha function, a kernel rate equal to a propagator pole) gives
  the attached leg its own time `s` and becomes one more smooth edge `s → t`, every row a difference row.
  `DRESS_CONV_KERNELS` (env `DAEDALUS_PHASE_J_DRESS_KERNELS=0`) forces that path.
- **Duplicated legs.** A vertex whose kernel sits on a leg that occurs several times (`exp(Σ_j w_ij (r_ij ∗ n_j))`
  Taylor-expanded: every `n` leg carries `r`; `x·(g ∗ x)`: one plain and one kernel copy) now has one kernel slot
  per copy, read from the power of the kernel symbol in the coefficient. When the copies differ, the diagram's
  weight (which counts them as identical: the automorphism colouring does not mark kernel legs) multiplies the
  average over the distinct placements of the lines in the slots; a single placement was 34 % off on
  `x·(g ∗ x)`. A δ kernel (`Conv(c·δ, n) = c·n`) is a weight, no delay. A delayed step `heaviside(t − d)` or a δ
  mixed with a smooth part is refused with a named reason (its delay would have been lost).
- **Kernels declared by `freq_image` only.** Such a vertex had no time-domain kernel: its one loop came out exactly 0
  (or, on a warm cache, crashed with `'ConvVertexType' object has no attribute 'response_legs'`). The modes now come
  from the poles of the rational image (`λ = iω_α`, `C = i Res ĝ(ω_α)`, retarded poles); an image with a δ part, a
  repeated or an acausal pole is refused by name. Spike-only Hawkes with `freq_image`, `time_expr`, or either through
  `define_function`: one loop equal to the voltage form to 1.3e-14.
- **Repeated kernel rates are refused, not mis-summed.** A kernel with a repeated or near-repeated rate (the alpha
  kernel `t e^{-t/τ}/τ²`, `ĝ = 1/((1+iωτ)(1+iωτ₂))` with τ₂ near τ) has `t e^{μt}` terms no mode sum holds; a
  numerically split double root gave a one loop 7x too large with status 'ok'. Rates closer than 10 % (relative) or a
  kernel that is not a finite exponential sum are refused by name (write the filter as auxiliary linear fields);
  between separated rates the result equals the Markov embedding to 1.6e-13 at an 11 % gap.

#### Fixed: the propagator and typed-diagram caches served another model with the same name (`api/_propagator.py`, `api/_diagrams.py`; pre-existing)

- Both caches are keyed by the model name; a model that differs only in a kernel's image (or any other text) got the
  other model's propagator: e.g. an alpha-kernel model returned an exponential-kernel model's numbers exactly. Both
  now store the model's `spec_signature` (action, equations, functions, kernel images) and rebuild on a mismatch. A
  cache file written before this change has no signature and is still accepted (clear `saved_models/` to be sure).
- Checked: the same models with the kernel written as an auxiliary linear field (`τ_g z' = y − z`, a local `x·z`
  vertex, none of this code) agree at tree level exactly and at one loop to 1e-12 (k = 2, auto- and
  cross-cumulants; also the mixed `x·(g ∗ x)` vertex); the one-loop δ weight equals the k = 1 one-loop mean to
  2e-16; a long simulation of the spiking model agrees with tree + one loop (the loop is 13 % of the tree there;
  the residual is the size of a two-loop term). Spiking model above: one-loop `C(tau)` at τ = 2.5 was 1.71e-6
  (8/29 diagrams, warm) or did not finish (cold); now 4.17e-6, 2.9 s. An exponential Hawkes network written
  spike-only (every leg carries the two-exponential kernel `H_i ∗ g_ij`) equals its voltage form: one loop to
  4e-14 relative (k = 2, 1446 diagrams in 35 s, where it did not finish in 12 minutes before), the k = 1 rate
  correction to 7e-15, k = 3 at tree level to 5e-14.
- No tracked model has such a vertex; no tracked number moves.

#### Fixed: a diagram Phase J cannot evaluate is no longer dropped silently (`api/compute.py`, `api/_grouped_phase_j.py`, `api/analyze.py`)

- Before, such a diagram was listed in `phase_j_by_ell[ell]['skipped_kernel_ids']` and left out of `C_tau`, and
  `compute_cumulants` said nothing (on the grouped route, a diagram that failed inside the per-diagram fallback was
  not even listed). Now every route (per-diagram, grouped, the pole-cluster deformation) sets
  `result['status'] = 'incomplete'`, names the diagrams and the reason in `result['phase_j_incomplete']`, warns with
  `PhaseJIncompleteWarning`, and the analyzer adds `G6_DIAGRAMS_SKIPPED`; `policy={'mode': 'strict'}` raises
  `PhaseJIncompleteError` (with the result attached). A complete Phase J run has `status == 'ok'`. The warning is
  shown under `sage -python` too (whose launcher ignores warnings), a saved file of an incomplete run carries a
  `phase_j_status` stamp (`load_provenance`), and `dd.run`'s moment assembly marks its result incomplete when one
  of its extra runs is.

#### Added: high-precision exact route for ill-conditioned subsets (`dbm_integral.integrate_exp_sum_mp`)

- A DBM closed form that fails its conditioning test (its expansion cancels: near-equal decay rates) is evaluated
  once more by the same elimination in 40-digit arithmetic before the subset may go to quadrature
  (`USE_DBM_HIGH_PRECISION`, env `DAEDALUS_PHASE_J_DBM_MP=0` to switch off; counters `dbm_mp_attempted` /
  `dbm_mp_answered`). The elimination is the same code with its arithmetic passed in (a private mpmath context;
  the double path is unchanged and thread safe). It answered the 9 subsets of the spiking model that needed 411 s
  of quadrature before the kernels were folded into lines (they now need neither).

#### Fixed: a declared function applied to an expression, and `time_expr` kernels with symbolic time constants (`api/model_compiler.py`, `api/model.py`)

- `define_function('phi', ['v'], 'a[i]*exp(v)')` called as `phi[i](Em[i] + sum(w[i,j]*Conv(r[i,j], n[j]) for j in E))`,
  or `g(a*x)` in an SDE drift, or any function whose formal argument names no field (`define_function('g', ['v'],
  ...)` with field `x`), stopped at the mean-field sanity check with an unresolved `phi_1(...)`. The expansion only
  has Taylor renames for `phi_<i+1>` expanded at the saddles of its formal arguments (or at 0). A call whose
  arguments at zero fluctuation are neither is now inlined (the function body evaluated at the arguments) in the
  action and in the mean-field closure, and such a single-call mean-field equation is a compound saddle, as it is
  with the body written out. A bare call `phi(...)` inlined in a population of more than one element whose body
  reads `i` is refused with a message asking for `phi[i](...)`. Result: the `define_function` form and the hand-written form give the same action,
  saddle, propagator and cumulants, bit for bit (TemporalModelBuilder and SDE front ends). Calls of a function on
  its own field (`phi[i](v[i])`, `g(x)`) keep the formal path: every zoo model is unchanged (action, saddle,
  cumulants; in-process A/B against the previous compiler).
- A kernel given only by `time_expr`, a causal sum of `t^n exp(-a t) heaviside(t)` terms whose rates Sage cannot
  order (`(exp(-t/tau) - exp(-t/taug))/(tau - taug)*heaviside(t)`), came back as an unevaluated `integrate(...)`
  and the analyzer stopped with `G3_NONRATIONAL_KERNEL`. When Sage's integral does not evaluate, the transform is
  now taken term by term in closed form, `c t^n e^{-a t} -> c n!/(a + i omega)^(n+1)`, with `a` required to be
  positive from the parameter domains (`domain='positive'`), then factored: the image is the same expression as
  the hand-written `freq_image` (and regular at `tau = taug`), cumulants bit-identical. No assumption is made or
  left behind. Where the rule does not apply, the error names the parameter(s) without `domain='positive'` and
  points to `freq_image=`. Kernels Sage already transformed keep Sage's image.

#### Changed: documentation (no code path changes)

- README: the config-file section now says `daedalus run config.yaml` (the console script of `[project.scripts]`), not `dd run`, which is the name of the Python alias, not a command; `sage -python -m daedalus run ...` is unchanged. The same fix is in the module docstring and the missing-PyYAML message of `api/config_run.py` and in a test docstring.
- ARCHITECTURE.md lists the modules added since 0.2.0 (`api/{config_run,sde,noise,cumulant,cumulant_plot,symbolic_out,_pole_clusters,_expand_cache,colored_to_markovian}.py`, `engine/integration/time_domain/{expdd,boxfree,dbm_integral}.py`) with one line each.
- Tracked text no longer names models that are not in the public repository: the two perf tables and one sentence below that named a local two-field OU quartic model now describe it structurally, and `docs/spatial_pipeline.md` points at the public model and simulator paths (it still said `theories/...` and `models/...sim.py`).

#### Added: notebook CI workflow (`.github/workflows/notebooks.yml`)

- A separate workflow that executes every `notebooks/examples/**/*.ipynb` with `jupyter nbconvert --execute` (per-notebook and whole-job timeouts) in the same conda environment as `tests.yml`. It runs only on `workflow_dispatch` and a weekly schedule, writes the executed copies to a temporary directory (the shipped outputs are not touched) and uploads them as a build artifact.

#### Changed: `FieldTheory.expand` on non-polynomial actions, minutes to about a second (`engine/core/field_theory.py`)

- The action is now Taylor-expanded node by node: sums add, products and positive integer powers multiply with
  truncation at total degree `taylor_order` at every step, and each distinct non-polynomial atom (`exp`, `log`,
  a negative or fractional power, a function call) gets one `taylor()` of its own, memoised. Before, one
  `taylor()` of the whole action put every coefficient over the common denominator of all atoms (single
  coefficients of ~120 kB for a 2-field action with four sigmoids, 18 MB in all at order 4).
- `PolynomialRing(SR)` zero-tests every coefficient on construction and after each `+`, and asks Maxima
  whenever the coefficient holds a symbol with an assumption (`domain='positive'`). The zero test now first
  evaluates the coefficient at one point that satisfies every assumption (128-bit intervals; only for
  coefficients made of exact constants and operators with rigorous interval arithmetic); only when that cannot
  show it nonzero is Sage's own test used, so the verdicts are Sage's. Sectors and `S_raw` are built once
  from dicts, in the same dict order as before. The expand-cache loader uses the same test (format and keys
  unchanged).
- Timings, `[1/7]` (2-field action with four sigmoids in drift and state-dependent noise): taylor_order 2:
  108 s to 0.7 s; 3 and 4: not finished in 10 min, now 0.8 s and 1.2 s; the whole k = 2, max_ell = 1 run takes
  12 s. Loading its cached order-4 expansion: 100 s to 0.1 s. `dendritic_quad_soma_sigmoid`: 76 s / 344 s
  (orders 2 / 4) to 0.4 s. No zoo model got slower (worst new/old ratio 0.59).
- Output: on every polynomial or `exp` / formal-function model of the zoo the expanded action is the same ring
  element as before (coefficients trivially equal, same dict order), so their results do not move (checked bit
  for bit on ten zoo entries). On the sigmoid models (`dendritic_quad_soma_sigmoid` among the public ones) the
  coefficients are the same functions in a shorter form (equal to 1e-59 at random points) with another monomial
  order; `dendritic_quad_soma_sigmoid` k = 2, max_ell = 1 moves by 7.5e-16 relative.
- `DAEDALUS_EXPAND_LEGACY=1` restores the previous expansion and zero tests. The verbose action printout no
  longer prints each ring element to test it for emptiness (`str()` of a ring element over SR compares every
  coefficient with 1 and could take 10 s; `api/compute.py`).

#### Fixed: the SDE front end refuses drifts with no syntactic linear term (`api/sde.py`; no number moves for a model that built before)

- `SDE.drift('x', '-x/(1+eps*x^2)')`, `'-x*exp(-eps*x^2)'`, an SGD logistic-loss gradient, or any drift
  whose linear coefficient contains a field, raised `SDEError: ... rhs has no linear term ... field-independent
  coefficient`. The mean-field equation needs `lhs|_{Dt=0} != 0`, and the front end only knew how to move a
  syntactic linear term onto the lhs. Where there is none it now adds `c*x` to both sides of the mean-field
  equation (lhs `Dt*x + c*x`, rhs `drift + c*x`, c = 1). The residual is unchanged and so are the saddle
  solve (a root solve on the residual, in which `c*x` cancels) and the linear stability (`d(lhs - rhs)/dx`); the
  action is not touched. A note in `SDE.notes` records the step.
- Measured on seven models (rational, exponential, shifted rational, two coupled fields, Stratonovich
  multiplicative noise, and an SGD logistic-loss gradient without and with an L2 term): the six that take the new
  branch give the saddle, the propagator and the k = 2 tree + one-loop `C(tau)` bit for bit equal to the
  hand-written `(Dt + 1)*x` model, and a second c (5/2) moves nothing beyond 1e-12 relative. The L2 case keeps its
  old branch (c = the L2 rate) and agrees to 1e-12.
- A model that was refused at build time now builds and is judged by the analyzer: `drift('x', '-x^3')`
  (zero slope at its saddle) gives `G2_MARGINAL_MODE`, `x/(1+x^2)` with `stability_analysis(True)` gives
  `G2_UNSTABLE_SADDLE`, both from the true slope of the drift and not from c.
- Models that built before take the old branch: their equations, action and notes are unchanged.
- Still refused, with the workaround spelled out: a lhs or rhs that cannot be read symbolically (e.g. a
  population sum in the rhs), and a drift that does not depend on the field at all.

#### Added: symbolic tree-level covariance in the time domain, any denominator degree (`api/symbolic_out.py`, stage C2)

- `tree_modes(C0, omega)` / `TreeCovariance.modes(a, b)` return `C(tau)` as a
  `ModeSum`: terms `Mode(coeff, rate, power, side)`, each
  `coeff * tau**power * exp(rate * tau)` on `side` 'positive' (tau > 0) or
  'negative', plus the `contact` weight of delta(tau). Repeated poles give
  `power >= 1` terms (confluent residues). Residues are computed once per
  irreducible denominator factor in K[x]/(p) (K = the parameters' rational
  functions), so any degree works: roots of degree <= 2 are written out,
  higher ones are `RootOf` symbols (`ModeSum.roots`). `evaluate(tau, values)`
  and `numeric_terms(values)` work at 256-bit precision (`evaluate` also sums
  the modes at that precision, so nearly coincident poles do not lose digits).
  A value for a name that entered the construction (substituted with
  `values=`, or used by the mean-field solve) must match it, else an error.
- Stated assumptions replace the sampled half-plane decision: each result
  carries `assumptions`, sign conditions from the Routh array (a Sturm chain
  for even factors) that fix the split of the poles between the two half
  planes, and distinctness conditions (discriminants, resultants). They are
  checked whenever the modes are evaluated; outside them `AssumptionViolation`
  is raised. Parameter sampling is kept only as a cross-check of the Routh
  count against numeric roots.
- `integrate_against(C0, Kernel)` / `ModeSum.integrate_against` /
  `TreeCovariance.integrate_against`: `int L(s) C(s) ds` in closed form for
  exponential and polynomial-times-exponential kernels, including
  `Kernel.two_sided_exponential(a_pos, tau_pos, a_neg, tau_neg)`. Sums over the roots of
  a one-sided factor are traces, so the result is an exact rational function
  of the parameters at any degree. The delta(tau) term is included when the
  kernel is continuous at 0; otherwise `contact=` must say which value of
  L(0) to use.
- `pair_with_kernel(..., any_degree=True)`: the Fourier-route pairing without
  the degree limit (default unchanged).
- `mean_field='solve'` and `fix_mean_field` substitute the model's
  `mean_field=True` parameters (with population indices) instead of every
  name containing "star".
- Checked against the pipeline's tree level (`C_tau`, max_ell=0) to 1e-12
  relative on OU, an underdamped OU, a feedback filter chain with an
  irreducible cubic denominator, an equal-rate chain (double pole), and the
  linear Hawkes models (contact = mean rate), each also after positive
  domains were put on the same symbol names. No pipeline number changes.

#### Added: regime advisories and a one-table report in the model analyzer (D3, D4)

- New gate G5 in `api/analyze.py`, read-only (no number moves; added preflight time is a few ms). `ModelTraits`
  gains `pole_rate_min`/`pole_rate_max` (decay-rate range of the poles), `tree_std` (tree-level standard
  deviations), `loop_parameter_estimate` (`g_eff`, with `loop_parameter_detail` naming the dominant vertices) and
  `taylor_radius_ratio`. `g_eff` estimates the size of the one-loop correction relative to the tree from the vertex
  couplings, the propagator and the noise amplitude, without running a loop (definition in the docstring of
  `api.analyze._g_eff`; for `eps x^3` with OU noise it is `3 eps D / mu^2`, which equals the measured
  |one-loop / tree| at tau = 0). New codes: `G5_LOOP_PARAMETER` (warning, `g_eff > 0.2` and loops requested),
  `G5_TAYLOR_RADIUS` (warning: the Taylor series of a non-polynomial response about the saddle does not shrink fast
  enough at 3 tree standard deviations), `G5_ADVISORY_SKIPPED` (info). `G3_STIFF`, `G3_CLOSE_POLES` and
  `G3_REPEATED_POLE` now carry remedies. `mode: strict` never promotes the two G5 warnings to errors (advisories about the answer, not model defects).
  Tracked models whose default regime is flagged: the two quartic double wells (`g_eff` 0.30) and
  `ou_quartic_colored` (`g_eff` about 10).
- `ModelReport.table()` / `str(report)`: one plain-text table of at most 100 columns (traits that matter, route,
  pole statistics, `g_eff`, cost when known, findings by level); `dd.run(..., verbose=True)` prints it.
- `dd run config.yaml --dry-run` prints the resolved config and the report table (as YAML comments); new config
  keys `question.chi_grid` (alias `spatial_grid`) and `output` (directory or file stem); list-valued parameters
  are checked against the model's population sizes; a 3-element `tau_grid`/`chi_grid` list that looks like a range
  is an error (as in `dd.Config`); console script `daedalus` (`[project.scripts] daedalus = "daedalus:main"`); an inline
  `sde:` block can replace `model:` (single scalar populations; noise types of `api.noise`).

#### Added: the δ components of a cumulant (no `C_tau` number moves)

A shot-noise model (a point process: Hawkes, spike reset, δ spikes) has
spikes `w δ(...)` in its cumulants, on hyperplanes of the external times (for
k = 2 the point τ = 0 of an autocorrelation, weight = the mean rate). They stay
OUT of `C_tau` (unchanged: `C(0)` is the continuous part) and are now exposed:

- `dd.cumulant(result)` (`api.cumulant.Cumulant`): the smooth part per loop
  order (`smooth_by_ell`, the `C_tau_by_ell` arrays) and `deltas`, a list of
  `DeltaComponent` (`weight`, `location`, `equality`, `loop_order`,
  `diagram`, `provenance`; `weight_at(times)` for a k >= 3 hyperplane, whose
  weight depends on the other times); `total_on_grid(tau, mode='bin')` adds
  each spike as `weight / Δτ` to the grid bin holding it (what a binned
  simulation estimator sees; a spike on a bin edge is split half / half),
  `mode='exclude'` is `C_tau`; `compare_to_sim(sim)`; `spectrum(omega)`
  = Fourier transform of the smooth part (QAWF quadrature of the Phase J
  callables) + `Σ w` (the flat floor); `to_frame()` / `deltas_frame()`.
  `modes` is the documented slot for the exponential-mode representation
  (`ModeTerm(coeff, rate, power, side)`), not filled yet. Nothing is
  recomputed.
- `dd.plot_cumulant(result, show_delta=True, delta_style='arrow'|'bin')`
  (draws this plot instead of the default one): k = 2 arrows with the weight per loop order, or the bin convention (compare a
  binned simulation with 'bin'); k = 3 hyperplanes as lines in
  (t_2 − t_1, t_3 − t_1) with their weight density.
- `output_npz` / `output_csv`: a `delta_components` table (`delta_weight`,
  `delta_location`, `delta_ell`, `delta_diagram`, `delta_equality`,
  `delta_provenance`, `delta_complete`; CSV: one JSON line per component in the
  header). Files without it still load (`Cumulant.from_saved` then reports
  `complete=False`). The flag stamp gains `USE_SHOTNOISE_INTEGRATED`.

#### Fixed: δ records that were dropped (records only; `C_tau` is bit-identical)

- Loop corrections to a δ weight were skipped: a shot-noise subset that keeps
  internal times after the δ elimination (e.g. the one-loop mean rate under
  the spike) is now integrated by the smooth-subset evaluator and returned as a
  record (`integrated=True`). Spike reset (zoo parameters), ℓ = 1:
  Σ weights = −0.0379870525 = the k = 1, ℓ = 1 mean (1e-15 relative; same
  for the quadratic-rate Hawkes `single_population_quad_exp_test`); a
  simulation confirms the corrected weight (rate 4.1696 ± 0.0018
  against 4.1709 through ℓ = 1 and 4.2089 at tree level). The same route recovers k = 3 tree-level δ(t_i − t_j) records.
  `DAEDALUS_SHOTNOISE_INTEGRATED=0` restores the old records.
- δ of codimension > 1 (t_1 = t_2 = t_3 at k = 3, weight = the rate) were
  dropped; they are in the new `delta_contributions_multi`.
- `use_grouped_phase_j=True` dropped every δ record (with a
  `GroupedDeltaRecordsDroppedWarning` since 0.2.0); it now carries the
  per-diagram fallback's records (equal to the per-diagram path's).
- Each Phase J result reports `n_delta_subsets_dropped` (shot-noise subsets
  that still give no record); `Cumulant.complete` is False when any did.

## 0.2.1 (2026-10-13)

A patch release on top of 0.2.0. Its reason: near-degenerate and repeated
poles of the propagator gave silent garbage in 0.2.0 (and before); they are now
resolved or flagged. It also carries the first-use speed-up of the shipped
prediagram cells, `dd.build_cache`, and the model analyzer wired into
`compute_cumulants`. No Phase J number moves except where a pole cluster was
ill-conditioned (listed in the first entry).

### Fixed: near-degenerate and repeated poles of the propagator (numbers move only where a pole cluster was ill-conditioned)

Phase J writes each propagator entry as a sum of pole terms. When poles come
close, the residues grow like 1/gap and the sums cancel: about
E log10(1/gap) digits were lost (E: the propagator lines that see the
cluster), and nothing said so. At gap 0 (a repeated pole, a Jordan block) the
pole-residue form cannot represent the propagator at all. The exact tier also
took the poles from double-precision roots of Q(omega), off by about eps/gap
next to a cluster, which moved the residues by eps/gap^2. Example,
`ou_quartic_colored` (mu = 1, eps = 0.02, D = 1, tauc = 1/(1 + gap)), C(0):

| gap | tree before → after | one-loop before → after (exact) |
|---|---|---|
| 1e-2 | 1.004975124 → 1.004975124 | -0.0907470 → -0.0907470075 (-0.0907470075) |
| 1e-4 | 1.000049949 → 1.000049998 | -1.5 → -0.0900074997 (-0.0900074997) |
| 1e-7 | 1.40625 → 1.000000050 | 0 → -0.0900000075 (-0.0900000075) |
| 0 (tauc = 1/mu) | -2.1e15 → 1.0 | -1.2e29 → -0.09 (-0.09) |

`compute_cumulants` now finds pole clusters (poles within 5 % relative,
single linkage) whose residues cancel by more than a factor 10 (scaled by the
distance to the other poles and to the stability boundary), or repeated
poles, and averages Phase J over a deformation of the propagator that splits
each such cluster on a circle. The result is an analytic function of the
deformation parameter, so its Cauchy mean over the unit circle is the exact
value; the trapezoid rule converges geometrically (12 to 16 nodes, N/2 + 1
Phase J runs for a real system). The integrators are unchanged. The
residues are computed to 50 digits on this path. `result['pole_clusters']`
reports the clusters, the node count and an accuracy estimate (the Fourier
tail of the node values); `DAEDALUS_POLE_CLUSTERS=0` switches the
deformation off. Measured against exact references for gaps 1e-2 to 1e-10
and 0: the colored OU within 1e-14 (tree) and 5e-13 (one loop); three
clustered poles, including a triple root, within 2e-14 and 1e-11.

Never silent: a cluster that cannot be resolved (the exact tier failed, the
cluster sits too close to the stability boundary to be split, or the switch is
off) is reported in `result['pole_clusters']` with a lost-digit estimate,
which enters the G6 accuracy estimate; above 10 digits (an error above 1e-6)
it also issues a `PoleClusterWarning` and the analyzer warning
`G3_POLE_CLUSTER_UNRESOLVED` (an error under `policy={'mode': 'strict'}`). Phase J
refuses the nominal residues of a repeated pole if they are passed to it
directly. `G3_CLOSE_POLES` uses the 5 % linkage and is `info`, like
`G3_REPEATED_POLE` (both are resolved), so strict mode now refuses only the
clusters the pipeline cannot resolve (before: every pair within 1e-3 and every
repeated pole).

Numbers moved (public models): `multipopulation_test` (k = 2, ell <= 1,
nE1-nE2, the zoo parameters): tree at tau = 0, 2.5, 10 from 2.17769902883e-4,
1.53898833295e-3, 1.43719729879e-3 to 2.17800651238e-4, 1.53902329574e-3,
1.43721885568e-3 (1.4e-4, 2.3e-5, 1.5e-5 relative). Its degree-16 Q(omega) has
two clusters; the old double-precision roots there were 4.5e-5 off. The new
value equals the old pole sums fed with 50-digit residues to 6e-16. At their
default parameters (k = 2, tree level, tau = 0, 2.5, 10 and 0, 2.5):
`quadratic_hawkes_alpha` (n1-n2; a pair, kappa 42) moves by up to 6e-12
relative (8.54678667117e-3 → 8.54678667122e-3 at tau = 2.5), and
`dendritic_quad_soma_sigmoid` (nS1-nS2; four poles within 5 %, kappa 2.7e4)
by 1e-9 (3.60326581144e-4 → 3.60326581512e-4 at tau = 0, 3.24664431617e-5 →
3.24664431924e-5 at tau = 2.5); both equal the 50-digit pole sums to 1e-15.
`ou_quartic_colored` moves only near tauc = 1/mu (table above). Every other
model of the zoo is bit-identical (in-process A/B against the previous
sources).

Cost: N/2 + 1 Phase J runs where a cluster is resolved (multipopulation_test
1.9 s → 3.6 s; ou_quartic_colored near tauc = 1/mu 0.2 s → 0.3 s; k >= 3
callables are evaluated once per node), none elsewhere.

Also fixed: the exact tier refused floats that Sage keeps as complex doubles
with a zero imaginary part (for example 2.0 from a product of imaginary floats
in a kernel image) and silently used the numerical tier. No zoo model was
affected.

### Changed: first use of a shipped prediagram cell is about 3.5 times faster (no numbers move)

The first use of a shipped cell on a fresh clone rebuilds every class from its
certificate, and that rebuild was almost all of the time. `record_from_cert`
now works out the degrees, the leaves-first relabelling and the sorted edge
list in plain Python and builds each of the two graphs once (it used three
Sage graphs and a `degree` call per vertex). The record is the same, bit for
bit: same vertex lists and order, same labelled edges in the same iteration
order, same `leaves` and `internal`. The old code stays as
`_record_from_cert_reference`, the oracle of `tests/test_record_from_cert.py`.
Rebuild of a whole cell from an empty directory, minimum of 3 runs (cell, old
→ new): (2,3) 0.93 s → 0.27 s; (3,2) 0.29 s → 0.095 s; (4,2) 4.75 s → 1.37 s.
Peak memory is unchanged (the record list itself dominates: about 11 kB per
class).

### Added: `dd.build_cache`, a cost report and a build for prediagram cells

`dd.build_cache(cells, dest=None, dry_run=False)` says, before doing any
work, which of the requested `(k, ell)` cells are shipped, already under
`dest`, or missing; the class count (from the manifests, else extrapolated
from the smaller cells of the same `k`); and for each missing cell the
estimated build time and memory, plus the memory the records of that cell take
once used (about 11 kB per class). The time comes from a measured rate
(25-70 microseconds per class, one process) scaled by a 0.15 s timing of a
small cell on the machine. `dry_run=True` stops there and writes nothing;
otherwise the missing cells are built by the existing streaming enumerator and
written as stamped v2 files. A build whose estimated memory exceeds the
machine's is refused unless `force=True`. For the record, a cold build of
(2,4) (1,152,032 classes) takes 80 s with one process on the compiled
enumerator and about 0.2 GB for the certificates; using it afterwards needs
about 13 GB for the records.

### Added: the model analyzer runs inside `compute_cumulants`; backend registry (no number moves)

- `compute_cumulants` (and `dd.run`, `dd run config.yaml`) runs the analyzer
  gates of `api.analyze` at its stage boundaries, all before the diagram
  enumeration: declarations and structure before the expansion, unpaired
  fields after it, the propagator class after the propagator, the mean field,
  noise and undriven fields after the saddle solve, the pole statistics after
  the residues. A finding of level `error` stops the run with
  `api.analyze.AnalyzerError` (a `ValueError`) naming its code and a remedy.
  The gates read the pipeline's own objects and change no number:
  results are bit-identical with the analyzer on or off on the whole model
  zoo. New keywords: `policy` (an `api.analyze.Policy` or dict: `mode`
  `'strict'`/`'permissive'`, consents, and `downgrade`, the error codes
  accepted as warnings), `preflight` (`False`, or `DAEDALUS_PREFLIGHT=0`,
  turns the gates off), `backend` and `accuracy`. A bug inside a gate is
  recorded as a warning and never stops a run.
- Backend registry (`api.routing`): each integrator backend (spatial
  integrator, grouped and per-diagram Phase J, the symbolic Phase H library)
  declares `applies(request) -> (bool, reasons)` and a priority; the router
  takes the first that applies. The built-in table reproduces the 0.2.0
  dispatch exactly, including its error messages and exception types; a new
  backend is one `routing.register(...)` call. When none applies, the error
  names the first failing trait and lists every rejected backend.
- Provenance: the result carries `result['route']` and `result['analysis']`
  (the analyzer report); `save_npz` / `save_csv` stamp `route`,
  `assumptions` (the assumptions ledger) and `analyzer_findings`;
  `api.load_provenance` reads them and returns `None` for files written
  without them.
- After the run (G6), the report carries the Phase J convergence-certificate
  verdicts, the error-accounting counters and a total-relative accuracy estimate
  (the summed rounding-error estimates of the closed-form subset integrals
  against the cumulant over the τ grid; a warning above the per-subset
  acceptance tolerance 1e-10, or above `accuracy` when given). Two
  bookkeeping counters (`dd_err_abs_sum`, `dbm_err_abs_sum`) and the record
  `prop['pole_analysis']` feed it.
- Analyzer fixes: smooth colored-noise kernels are evaluated with their
  registered signature (they were all reported as non-rational) and a
  non-rational one is a warning, since Phase J computes it by its quadrature
  fallback; G3 for spatial models (growing finite-wavenumber modes, negative
  diffusion, massless fields); the expand-cache validity check is one
  function shared by `load_expand` and the analyzer.

## 0.2.0 (2026-10-12)

### Changed: Phase J integrates exactly, without an integration box, by default (numbers move)

The box-free exact routes (see "Added: box-free exact Phase J" below) are
now the default: `USE_PHASE_J_DD_KERNELS`, `SEMIINF_CHAIN` and
`UNBOUNDED_2D = 'exact'` are on unless the environment says otherwise. Every
Phase J region is then integrated in closed form on the divided-difference
kernels, with no ±200 box, no margin-50 lower limit for chains and no
`scipy.nquad` fallback on any model measured. Unset, the two box-free
switches follow `DAEDALUS_PHASE_J_DD_KERNELS`.

How to get the previous behaviour:

- `DAEDALUS_PHASE_J_DD_KERNELS=0`: the boxed routes of 0.2.0 before this
  change (the Θ(0) = 0 rule, the inheritance fix and the hardened fallback
  stay);
- `DAEDALUS_PHASE_J_LEGACY=1`: the pre-0.2.0 routes and numbers (it resets
  every Phase J switch, the box-free ones included);
- in a running process, `final_integral._legacy_umbrella_flags()` lists every
  flag the umbrella sets and its value.

What moved (the public models of the zoo, new code, in one process, boxed
routes → box-free; more detail under "With the switches on" below):

| model (configuration of the zoo) | change |
|---|---|
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | one-loop +6.3 % at τ = 0, +3.0 % at τ = 2.5, +2.7 % at τ = 10 (total up to 3.4 %); the boxed chains truncated its slow modes. 48 `scipy.nquad` regions → 0 |
| `single_population_spike_reset_test`, k = 2, ℓ = 1, per-diagram and grouped | ≤ 5.8e−13 relative; 40 and 10 `scipy.nquad` regions → 0 |
| `ou_quartic` family, `ou_quartic_colored`, `ou_quartic_two_dim_color_corr` | ≤ 1.1e−13 relative |
| `linear_hawkes`, `multipopulation_test`, `single_population_linear_delta_spikes_test`, spike reset k = 1 | ≤ 3.2e−16 relative |
| spatial models (`allen_cahn_1d_subcritical_infinite`, `reaction_diffusion_2d`, `coupled_rd_2species_1d`, `reaction_diffusion_conserved_1d`, `edwards_wilkinson_1d`) | identical; spatial certification unchanged |

The frozen regression fixture `spike_reset_k2_ell1` was refrozen once on the
new defaults (its strict xfail is lifted): its one-loop now agrees with the
ħ-extrapolated simulation's loop coefficient (per-bin χ² 5.8 over 10 bins,
p = 0.83). The other three fixtures did not move (≤ 1.5e−15).

Other effects of the new default:

- Speed: with the batched kernels (see "Performance: batched box-free
  kernels") the default runs are as fast as or faster than the boxed ones on
  every model measured (public zoo models, Mac): `quad_exp` 25.2 s → 18.8 s,
  the rest unchanged within noise.
- A non-finite residue or exponent now raises `ValueError` on the default
  routes ("never a silent number"); the boxed routes return NaN, as before.
- A region whose open direction does not decay is reported and handled by
  the divergence policy (see "Added: convergence certificate verdicts and a
  divergence policy"); none occurs on the models above.
- `CHAIN_UPPERS_KERNEL` stays `'legacy'`: the box-free chain route evaluates
  its chains with `expdd` and never reads it.
- Fixed with it: with the box-free routes on and `USE_CHAIN_UPPERS_MEMO`
  off, the chain sums went through a different summation than with the memo
  on and differed in the last bits; the memo now changes no number on any
  route.
- The streamed-vs-eager gate of `TEMPORAL_CACHE_OFF_STREAMS`
  (`test_temporal_streamed_totals_match_eager_on_a_fallback_model`):
  `single_population_quad_exp_test` now agrees to 1e−13 relative too (it
  was a strict expected failure at 1.21e−11). The flag stays False: fewer
  regions reach the fallback, not none (`single_population_quad_exp_test`
  sends 8 three-time regions there at τ = 0.5 and 1, whose lower bound the
  box-free chain cannot inherit and the exact DBM declines as
  ill-conditioned; `quadratic_hawkes_alpha`, k = 2, ℓ = 1, one).


### Fixed: accuracy accounting and global state (numbers unchanged on the models measured)

- The exact difference-constraint route (DBM) counts the error of a
  coefficient β it integrates as 0 (|β| below its threshold) in its
  conditioning test: such an error could pass with status OK (8e−9
  relative on a constructed case); only the part of β beyond the rounding
  margin of the exponents is counted.
- The box-free routes apply the same conditioning test to every subset sum
  as bookkeeping: counters `dd_err_ratio_max` (worst estimated-to-accepted
  error ratio) and `dd_ill_conditioned` (subsets above 1). The values are
  used as computed. The test is relative to the subset: on the models
  measured only `quad_exp` flags any (8 regions near a cancellation, about
  1e−21 against a one-loop of 2e−4).
- `StaleResultWarning` and `PhaseJNquadFallbackWarning` are shown under the
  Sage launcher's `PYTHONWARNINGS=ignore` (as the grouped δ-record warning);
  filters set in code (`-W error`, `simplefilter('error')`, a targeted
  `ignore`) are respected.
- The SDE front end's mean-field kernel integral no longer forgets a
  positivity assumption that was held before it (a model builder's
  parameter domain), and skips one implied by a stronger held assumption.
- The divergence policy also applies to the divergences the boxed interval
  routes report (`DAEDALUS_PHASE_J_DD_KERNELS=0`); a two-time region
  containing a line is 0 when every pole-tuple group cancels exactly.
- The batched box-free kernels are compiled without numba's on-disk cache
  (a cached kernel kept stale code after an edit to `expdd.py`): one
  compilation, about 4 s, per process that uses them.
- Tests: a guard (`tests/conftest.py`) fails a test that removes a Sage
  assumption other code holds or leaves a symbol with contradictory
  domains; four tolerance lines that passed only through pytest's absolute
  floor now check what they state.

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
| a two-field OU quartic model (local), k = 2, ℓ = 2, 7 τ | 46.3–49.4 s | 28.1–29.2 s | 2,613,114 / 54,726 |
| `ou_quartic`, k = 4, ℓ = 2, 1 point | 36.8–38.4 s | 19.4–20.1 s | 473,592 / 40,248 |
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | 89.2 s | 42.0 s | 7,988,455 / 584,321 (cap reached) |
| `single_population_spike_reset_test`, k = 2, ℓ = 1 | 14.7 s | 12.3 s | 173,278 / 3,922 |
| same, grouped | 6.3 s | 4.6 s | 2,958 / 3,922 |

### Performance: per-diagram setup levers (numbers unchanged)

After the chain memo, most of the time of a run with many diagrams went into the
per-diagram setup of `integrate_diagram`: on a two-field OU quartic model (local; k = 2, ℓ = 2)
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
| a two-field OU quartic model (local), k = 2, ℓ = 2, 7 τ | 35.0 / 35.2 s | 10.8 / 11.2 s |
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

### Added: box-free exact Phase J on the divided-difference kernels (opt-in when added; the default since the change above)

Three switches route the Phase J closed forms to the divided-difference
kernels of `expdd` and remove the integration box from directions in which
the integrand decays. All three are off by default; with them off every
result is bit-identical to before (in-process comparison, same cache, on 38
model configurations, every public one listed under "With the switches on"
below among them).

- **`USE_PHASE_J_DD_KERNELS`** (environment `DAEDALUS_PHASE_J_DD_KERNELS=1|0`)
  routes the two-time polygon (triangles anchored at the vertex with the
  largest real exponent), the one-time interval (a first divided
  difference, so a nearly cancelling exponent costs no accuracy), the chains
  of three or more times with their intermediate upper bounds (transfer
  matrix; `CHAIN_UPPERS_KERNEL` is then not read) and the grouped
  zero-time evaluation to `expdd`, on the per-diagram and the grouped path
  alike. The constant part γ of each exponent is folded into the kernel's
  logarithmic scale before any exponential is taken. The two-sided
  |Re| > 600 guards of those paths are replaced by one test on the final
  exponent: an integral beyond e^700 is an overflow, and it is reported (the
  region goes on to the next route), never returned as `inf` or `nan`. (On
  the two-time and one-time routes the test is made on the whole region's
  sum, on the chains per ordering of the times.) The exact
  difference-constraint integrator closes open directions at
  ±max(200, 40/κ + the largest |external time|) instead of ±200, κ the
  slowest decay rate of the region's decaying modes.
- **`SEMIINF_CHAIN`** (`DAEDALUS_PHASE_J_SEMIINF_CHAIN=1|0`; needs the first
  switch) integrates a chain without a scalar lower bound from −∞ (the
  semi-infinite chain, started from the dominant eigenvector) instead of
  from the earliest external time − 50, and a chain whose top has no upper
  bound to +∞ (its mirror image) instead of to 200. The difference-
  constraint integrator then has no box at all: an open direction is
  integrated in closed form to infinity.
- **`UNBOUNDED_2D`** (`DAEDALUS_PHASE_J_UNBOUNDED_2D=box|exact`; `exact` needs
  the first switch) builds a two-time region exactly, from its finite
  vertices and the rays of its recession cone, with no ±200 box, and
  integrates it as a fan of triangles, half-strips and wedges
  (`engine/integration/time_domain/boxfree.py`).
- The two-time region is built in exact rational arithmetic, as every other
  emptiness test of Phase J: a region of measure zero is empty, a genuinely
  thin one is integrated (or found divergent), whatever its width.
- An open direction in which the integrand does not decay (a recession ray
  with Re(α·r) ≥ 0, a line in the region, a chain prefix sum with Re ≤ 0, with
  a margin of 64 rounding units of the exponents so that the rounding residue
  of an exactly constant direction never passes as decay) is never integrated
  over a box on these routes: it is reported (counter `divergent_subsets`)
  and the region goes on to the next route. The policy for such regions is a
  later change.
- **The finite lower bound stays a parameter.** `final_integral.PHASE_J_T_MIN`
  (default −∞) is the lower bound of every integration time on the box-free
  routes, in absolute time (each Wick permutation's time frame is taken
  into account), for later switch-on transients: with T = −3 the tree of
  `ou_quartic` at τ = 2 is e^−2 (1 − e^−6), the process started at 0 at
  T. A finite value needs all three switches, raises if an external time is
  before it, and raises if a region would reach the `scipy` fallback, which
  could not honour it. That happens when T is within a few decay lengths of
  the external times: the exact difference-constraint route can then
  decline a region next to T as ill-conditioned (spike reset: T = −1 and −2
  near external times 0 and −2 raise, T = −5 works); the error names the
  region, the external times and the remedy (move T further back).
- **`DAEDALUS_PHASE_J_LEGACY=1`** now also resets `USE_PHASE_J_DD_KERNELS`,
  `CHAIN_UPPERS_KERNEL` (to `'legacy'`), `SEMIINF_CHAIN` and `UNBOUNDED_2D`:
  the umbrella reproduces the pre-0.2.0 numbers, and the divided-difference
  kernels agree with the legacy closed forms only to rounding.
- **Grouped path and δ(τ) records.** `use_grouped_phase_j=True` does not carry
  the equal-time spike records that the per-diagram path returns in
  `phase_j_by_ell[ell]['delta_contributions']`. It now says so: one
  `GroupedDeltaRecordsDroppedWarning` per `compute_cumulants` run that
  dropped any (shown also where the environment ignores warnings, as the
  Sage launcher does with `PYTHONWARNINGS=ignore`; filters set in code, a
  filter for this category and `-W error` are respected), and
  `n_delta_records_dropped` in the grouped
  result. No number changes (neither path adds the records to `C_tau`).
- **Saved results** carry a third stamp, `phase_j_flags`: the switches and
  tolerances of `final_integral` and `grouped_integral` that can move a
  Phase J number, and `use_grouped_phase_j` (also in `result['config']`)
  (provenance only; `load_npz` does not check it).
- New counters: `dd_polygon`, `dd_interval`, `dd_poset`, `dd_grouped_m0`,
  `polygon_exact_unbounded`, `poset_semiinf_chains`, `dd_overflow`,
  `dd_unsupported`, `divergent_subsets`, `dbm_box_free`, `dbm_slow_mode_box`,
  `dbm_declined_divergent`, and for the legacy chain evaluator
  `chain_uppers_legacy_partial` (a value assembled from only some of its
  cut cases) and `chain_uppers_legacy_huge` (a finite value beyond e^700).

**With the switches on (all three), measured.** Every region of the
configurations below is answered by a closed form: the `scipy.nquad`
fallback is never called (before: 48 calls for `quad_exp` k = 2 ℓ = 1, 40 and
10 for spike reset per-diagram and grouped), and the legacy polynomial chain
routine is never called (before: 246,515 calls for `ou_quartic` k = 4 ℓ = 2).
No region was reported as divergent and none overflowed. For `quad_exp` and
spike reset (per-diagram and grouped), every region was also compared with
independent references (tight quadrature, a 50-digit fan formula and the
box-free difference-constraint integrator): all agree to the harness
tolerance max(1e−8·|ref|, 1e−14), except where a reference's own box
truncates a slow mode (see `quad_exp`); the others are compared as whole
results.

| model (configuration of the zoo) | change |
|---|---|
| `single_population_quad_exp_test`, k = 2, ℓ = 1 | one-loop at τ = 0 (−1e−6) +2.016417e−4 → +2.142877e−4 (+6.3 %), τ = 2.5 +3.777821e−4 → +3.892182e−4 (+3.0 %), τ = 10 +2.955943e−4 → +3.035783e−4 (+2.7 %); tree unchanged. The old values truncated the chains of three and four times at the earliest external time − 50, which is not enough for this model's slow modes (decay rate 0.1). The new values equal, region by region, the box-free difference-constraint integrator (sum within 8.4e−16) and the 50-digit fan on a ±400 box; the ±150 and ±200 references are themselves truncated. The grouped path moves by up to 5.9 % to the same values |
| `single_population_spike_reset_test`, k = 2, ℓ = 1, per-diagram and grouped | ≤ 5.8e−13 relative (the regions the hardened `scipy.nquad` fallback served) |
| `ou_quartic` (k = 2 ℓ ≤ 3, k = 3 ℓ = 2, k = 4 ℓ = 2), `ou_quartic_colored`, `ou_quartic_two_dim_color_corr` | ≤ 1.1e−13 relative |
| `linear_hawkes`, `multipopulation_test`, `single_population_linear_delta_spikes_test`, spike reset k = 1 | ≤ 3.2e−16 relative |
| spatial (`allen_cahn_1d_subcritical_infinite` ℓ = 2, `reaction_diffusion_2d` ℓ = 1, `coupled_rd_2species_1d` ℓ = 1, `reaction_diffusion_conserved_1d` ℓ = 1, `edwards_wilkinson_1d`) | identical |

The test model with quadratic drift of `tests/test_phase_j_moment_oracle.py`
moves by 7.7e−9 relative at one loop for the same reason as `quad_exp`
(slowest decay rate 0.42, margin 50); the switches give the value of the
legacy path with the margin pushed to 150, to 1e−12.

The same truncations make the default path wrong, and the switches right,
whenever modes are slow or times are long (an adversarial review against an
exact multi-time oracle built from the generator, and per-region references;
the default-path error first): `ou_quartic_colored` at its defaults 1.2e−3;
`ou_quartic_two_dim_color_corr` at its defaults 2.5e−4; `ou_quartic` with
μ = 0.1, k = 2 ℓ ≤ 2, 4.3e−3; `ou_quartic` k = 2 at τ = 250, ℓ = 2 almost
entirely (ℓ = 1: 2e−3), and at τ ≥ 600 every loop order; spike reset with slow
rates 8e−4; `linear_hawkes` k = 3 tree 2e−6 to 7e−6. With the switches on
each of these agrees with its exact reference to ≤ 7e−14 (to ≤ 3.3e−15 on the
colored models), except the `linear_hawkes` k = 3 tree, which has no exact
reference and agrees with per-region quadrature references to 2e−12 to
7e−12. The default path also does not finish (killed after 6 minutes, in
the `scipy.nquad` fallback) on `ou_quartic` k = 4 ℓ = 2 at (0, 20, 20, 50),
which the switches answer in 4 s, exact to 4.6e−15. And the default two-time and
chain routes return 0 when exp(γ) underflows although the region's own
exponent compensates (fast rates, external times beyond about 70); the
switches do not.

The legacy evaluator of chains with intermediate upper bounds can be wrong
on adversarial inputs that the transfer matrix gets right. On the zoo
configurations (the public ones above, about 645,000 distinct calls), every
call was evaluated both ways: the largest disagreement is 2.9e−11 relative
(`quad_exp`), no call assembled a value from only some of its cut cases
(counter `chain_uppers_legacy_partial`), and none returned a value beyond
e^700. So those inputs do not occur there. They do elsewhere: the counter
fired at wide and very large τ (`ou_quartic` k = 2, ℓ = 2 and 3: 1 to 18 calls) and
on a coupled two-field model at k = 4 (260), where the totals still agreed
with the exact oracle to 3e−15; where it matters, the switches move the
number to the correct one.

Speed: with the switches on the kernels first ran per pole tuple in Python,
up to 2.3x slower than with the switches off on models with many poles per
region. The batched kernels (next section) make the switches-on runs faster
than the switches-off ones on every model measured.

### Performance: batched box-free kernels (switches on; numbers unchanged to rounding)

With the box-free switches on, the two-time regions and the chain folds of
three or more times now take all pole tuples of a region at once on the
compiled backend of `expdd` (plain Python without numba):

- `boxfree.region_integral_batch` / `region_modesum` integrate a two-time
  region for every pole tuple in one compiled loop (`multipopulation_test`
  k = 3 has about 67,000 tuples per region);
- the chain route keeps the chains of one ordering of the times for EVERY
  pole-tuple group of a region in one memo entry, keyed exactly on the bytes
  of the groups' exponent coefficients, the ordering, the bounds and the
  decay margin (filled from the per-chain memo, so the chain values are the
  same objects; bounded by 2,000,000 chains, cleared with the chain memo),
  and `boxfree.group_fold` / `chain_fold` fold the prefactors and sum the
  terms in one compiled loop. Before, a run made up to 160 million Python
  calls into the per-chain memo (`quad_exp`: 9.3 million calls for 586,000
  distinct chains);
- both take a region only when every term is plain (status OK, the
  in-range branch); anything else (a divergence, an overflow, a non-finite
  input) reruns the scalar loop, so statuses, bail reasons and errors are
  those of the scalar routines. Regions with fewer than 8 pole tuples (or
  pole-tuple groups) stay on the scalar loop, where the batch's fixed cost
  per call and per ordering would dominate (`ou_quartic` k = 4 ℓ = 2, one
  group and about 500 orderings per region, ran 1.4x slower batched).

Values agree with the scalar loops to rounding, not bit for bit (compiled
complex products may be contracted to fused multiply-adds): in one process,
switches on, the totals of the configurations below agree with the previous
code to ≤ 5.2e-16 relative, with the same counters (the chain-memo hit and
miss counts aside) and the same route and status for every region. Single
regions move by up to 2.8e-9 relative, all of them sums that cancel by a
factor of 1e7 to 9e14 (sum of |terms| over |sum|); there both codes are
within 1.5e-15 times that factor of the exactly rounded sum. With the
switches off every result is bit-identical.

Wall time, whole run, minimum of 3 (Mac, in one process per model, the
three variants interleaved):

| model (configuration) | switches off | on, before | on, now |
|---|---|---|---|
| `single_population_quad_exp_test` k = 2 ℓ = 1 | 25.3 s | 33.1 s | 18.8 s |
| `multipopulation_test` k = 3 ℓ = 1 (6 points) | 27.2 s | 99.6 s | 18.7 s |
| a coupled two-field model k = 4 ℓ = 1 (6 points) | 131.5 s | 253.1 s | 54.2 s |
| `single_population_spike_reset_test` k = 2 ℓ = 1 | 1.5 s | 1.3 s | 1.1 s |
| same, grouped | 2.6 s | 2.6 s | 2.5 s |
| `ou_quartic` k = 4 ℓ = 2 | 7.2 s | 6.5 s | 6.5 s |
| `ou_quartic` k = 2 ℓ = 3 | 0.8 s | 0.8 s | 0.8 s |
| `multipopulation_test` k = 2 ℓ = 1 | 1.8 s | 1.8 s | 1.8 s |

### Added: convergence certificate verdicts and a divergence policy for the box-free routes (default numbers unchanged)

The box-free routes (see "Added: box-free exact Phase J") certify
convergence before they integrate: every open direction of the region must
decay (an infinite side of an interval, a recession ray or a line of a
two-time region, the prefix or suffix sums of a chain). What happens when a
region fails it is now decided explicitly.

- **Exactly cancelling groups are dropped.** Pole tuples that share their
  exponent coefficients form a group; when the group's merged coefficient
  Σ C e^{γ − γ_max} is exactly 0 in floating point (for example a zero-rate
  mode with opposite residues), its integrand is identically zero and a
  failed certificate is not a divergence. The interval and two-time routes
  drop such a group (new counter `divergent_cancelled`); the chain route
  merges the groups before it evaluates a chain, now also on the grouped
  path's no-plan branch (before, a cancelling pair of zero-rate tuples
  there was reported as divergent).
- **Divergence policy.** Any other failure is a divergent region (counter
  `divergent_subsets`, bail `<route>_divergent`). The caller's policy then
  applies: `'fallback'` (default) takes the next route (the exact
  difference-constraint integrator for three or more times, then the boxed
  `scipy.nquad` fallback) and warns once per process with a
  `PhaseJDivergenceWarning`; `'warn'` does the same with one warning per
  region naming the diagram, the region, the external times and the route;
  `'raise'` raises `PhaseJDivergenceError` with the same details. The policy
  is a keyword `on_divergence` of `integrate_diagram`,
  `compute_correction_td`, `integrate_grouped_diagram` and
  `compute_correction_td_grouped`; `None` takes
  `final_integral.PHASE_J_ON_DIVERGENCE`, read at call time (environment
  `DAEDALUS_PHASE_J_ON_DIVERGENCE=fallback|warn|raise`; it changes no number,
  so `DAEDALUS_PHASE_J_LEGACY` does not touch it). The spatial certification
  passes `'fallback'` explicitly. The warning is shown also under the Sage
  launcher's `PYTHONWARNINGS=ignore`.
- `nquad_fallback_by_reason` has three new categories, `divergent`,
  `dd_overflow` and `dd_unsupported`, for the box-free routes' bails (they
  were counted under `other` / `polygon_other`).
- An unknown value of `DAEDALUS_PHASE_J_THETA0_CONST_ROW`,
  `DAEDALUS_PHASE_J_STRUCTURAL_ZEROS`, `DAEDALUS_PHASE_J_NQUAD_HARDENED` or
  `DAEDALUS_PHASE_J_DBM` now raises also when `DAEDALUS_PHASE_J_LEGACY=1` is
  set (it was ignored).

With the switches off (the default) no route reports a divergence, and every
result is bit-identical to before (in-process comparison); with the switches
on, the zoo configurations have no divergent and no cancelling region, and
their results are bit-identical too. The spatial certification is unchanged
(the massless probes fail as before, with the same error, in both).

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

- **Config-file entry point (`api/config_run.py`, `dd run config.yaml`).**
  `sage -python -m daedalus run config.yaml [--dry-run]` runs a YAML config:
  `model`, `parameters`, `question` (`k`, `max_ell`, `external_fields`,
  `tau_grid` as a list or `{start, stop, num}`, `points` for spatial k >= 3,
  `taylor_order`), `policy` (`use_cache`, `parallel`, `n_workers`, `verbose`)
  and `outputs` (`npz`, `csv`, `plot`, `report_pdf`). Validation is strict and
  every error names its config line; unknown keys are errors. `--dry-run`
  loads the model, checks the external fields against its physical fields and
  prints the resolved config without computing. A run calls `dd.run`, writes
  the outputs with the existing writers and saves `config_resolved.yaml` and
  `run_info.json` (version, git commit, resolved options, wall time) next to
  them. Relative paths resolve against the config file's directory. Numbers
  are unchanged: the totals equal a direct `compute_cumulants` call. The only
  edit to an existing file is a guarded `main()` at the end of `daedalus.py`.
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
  and `USE_DBM_FALLBACK = False`, and also `USE_PHASE_J_DD_KERNELS = False`,
  `CHAIN_UPPERS_KERNEL = 'legacy'`, `SEMIINF_CHAIN = False` and
  `UNBOUNDED_2D = 'box'` (see "Added: box-free exact Phase J"). It
  reproduces the
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
  - Nothing in Phase J calls the module by default. The first opt-in was
    `final_integral.CHAIN_UPPERS_KERNEL` (`'legacy'` by default, or
    `'transfer'`; environment `DAEDALUS_CHAIN_UPPERS_KERNEL=legacy|transfer`,
    any other value is an error; read at call time), which routes
    `_chain_with_intermediate_uppers` to `expdd.chain_with_uppers`. The
    selector is part of the chain-uppers memo key, so a toggle misses the
    memo. With `'legacy'` every result is bit-identical to before.
  - Flag `USE_PHASE_J_DD_KERNELS` (environment
    `DAEDALUS_PHASE_J_DD_KERNELS=1|0`, default 0, any other value is an
    error). It routes the Phase J closed forms to `expdd`: see "Added
    (opt-in): box-free exact Phase J" above, which also lets
    `DAEDALUS_PHASE_J_LEGACY` reset both switches.
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

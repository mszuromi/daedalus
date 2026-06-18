# Audit — `sections/20-simulators.tex` (Validation: Simulators & Cumulant Estimators)

**Verdict: accurate (minor-issues).** The chapter is an exceptionally faithful
description of `models/ou_langevin_sim_numba.py` and `models/cumulant_estimator.py`,
plus the wider `models/` supporting cast. Every named function, class, result key,
file, docstring quote, and line citation I checked resolves correctly. LaTeX is
valid (all 13 environment types balanced, all macros/tcolorboxes defined in the
`daedalus_manual.tex` preamble, zero bare specials outside math/listings, tikz
nodes all defined, longtable header well-formed). One genuine but minor factual
overstatement, plus two citation nits.

**LaTeX: OK** (`latex_ok = true`).

---

## Findings

- **minor** — *Manual claim:* the supporting-cast table says
  `dendritic_linear_sim_numba.py` "imports `compute_kpoint_slice`."
  *Code reality:* the file imports only `numpy` and `numba`
  (`models/dendritic_linear_sim_numba.py:47-48`); there is **no** import of
  `compute_kpoint_slice` or `cumulant_estimator` anywhere. The only reference is a
  **comment** (`:253`, "the `binned_counts` array fed to `compute_kpoint_slice`").
  The factual statement "imports X" is wrong; the true relationship is "its output
  is later consumed by `compute_kpoint_slice`."
  *Location:* `20-simulators.tex:1007-1008` (the `models/` table row).

- **nit** — *Manual claim:* "The oracle `coupled_box_correlator` uses two of its
  routines, per Fourier mode (`coupled_rd_1d_sim.py:73`)."
  *Code reality:* line 73 is the **import** statement
  (`from scipy.linalg import expm, solve_continuous_lyapunov`), not a per-mode
  *use*. The actual per-mode uses are at `:351` (`solve_continuous_lyapunov`) and
  `:353` (`expm`) — which the manual already cites correctly in the two bullets that
  follow. Citing the import line for "uses … per Fourier mode" is a slight mismatch
  of citation to claim; harmless because the precise use-site lines are also given.
  *Location:* `20-simulators.tex:949-960`.

- **nit** — *Manual claim:* the canonical wiring is "taken from **cell 9** of
  `notebooks/examples/temporal_ou_quartic_white.ipynb`."
  *Code reality:* the notebook exists and contains every quoted pattern
  (`fp = res['_resolved']['parameters']`, `float(fp['mu'])`, `sim_ou_quartic_numba`,
  the `# JIT warmup` call, `estimate_kpoint_slices(... voltage_bins=x_bins)`,
  `dt_bin_eff`, `C_tau_by_ell`), so the listing is faithful. I did not verify the
  exact ordinal "cell 9"; if cells were reordered this could be off by a cell. Not
  load-bearing.
  *Location:* `20-simulators.tex:1030-1031`.

---

## Spot-checks that PASSED (high-signal, non-exhaustive)

### `ou_langevin_sim_numba.py`
- Docstring action `S = ∫ dt xt·((Dt+mu)·x + eps·x³) − D·xt²` — `:29`. ✓
- EM step `x = x + dt_sim*(-mu*x - eps*x*x*x) + sqrt_2D_dt*np.random.randn()`
  and `sqrt_2D_dt = np.sqrt(2.0*D*dt_sim)` — `:74`, `:80-82`. ✓ (cubic written as
  `eps * x * x * x`, exactly as the manual notes).
- Bin accumulator `if cur_bin < n_bins: x_bins[0,cur_bin] = accum/bin_size_steps`
  with the full-buffer guard — `:84-89`. ✓ Output shape `(1, n_bins)`. ✓
- Sign conventions (mu>0 stable / mu=0 pitchfork / mu<0 wells at ±√(|mu|/eps)),
  stationary var `D/mu`, finite-eps shift `O(eps·D²/mu³)` — `:60-66`. ✓
- `dt ≪ 1/mu` rule — `:36`. ✓ `np.random.seed(seed)` `:68`, `np.random.randn()`
  `:82`. ✓ Signature `:17-21`. ✓
- Colored OU: `decay = np.exp(-dt_sim/tauc)`, `sigma_xi = np.sqrt((2D/tauc)·(1−decay²))`
  — `:146-148`. ✓ Step uses **previous-step** `xi`: `x_new = x + dt_sim*(drift + xi)`
  — `:160-166`. ✓ White-limit `⟨ξξ⟩ → 4D·δ`, `D_white = 2·D_colored` — `:127-132`. ✓
- Cholesky: `eta_y = rho*u + rho_perp*v` `:351-352`; clamp
  `if one_minus_rho_sq < 0.0: = 0.0` → NaN-not-error — `:332-335`. ✓
- Two-dim white-corr theory mismatch (`coefficient='D1'` = half-strength;
  edit to `'2*D1'` or halve D) — `:412-419`. ✓ (docstring "Convention note").

### `cumulant_estimator.py`
- `falling_factorial_array` body `result *= (n_arr - j)` — `:63-81`. ✓
- `compute_kpoint_slice` signature/def — `:84`; exactly-one-None sweep assert
  `:139-145`; field-type normalize + `_normalize_ft` (and the `1/dt_bin²` bug it
  fixes, "=16 for dt_bin=0.25") — `:148-202`, `:166-200`. ✓
- Helpers `_data_for`/`_is_spike_ft`/`_centering_denom` — `:204-221`. ✓
- Valid window `valid_start/valid_end/n_valid`, raise if `n_valid ≤ 2*max_lag_bins`
  — `:225-240`. ✓ Means on valid window — `:242-251`. ✓
- Fixed-leg product with `Counter` multiplicity, voltage-power-vs-spike-factorial
  asymmetry — `:261-300`, `:279-282`. ✓ Sweep series — `:302-307`. ✓
- FFT xcorr `ifft(F_product*conj(F_sweep)).real`, `n_fft ≥ 2*n_valid` next-pow-2,
  lag read `raw_xcorr[(-lag)%n_fft]`, the load-bearing **"Do NOT use [::-1]"**
  comment — `:309-320`, `:316-319`. ✓
- Lag-dependent norm `n_overlap = n_valid - abs(lag)` — `:322-332`. ✓
- k=4 cumulant subtraction κ₄ = m₄ − Σ pair-products; partitions
  `{(0,1)(2,3),(0,2)(1,3),(0,3)(1,2)}`; `_two_point` factorial-corrected iff
  same-pop ∧ same-spike-ft ∧ same-lag, else linear cov — `:334-433`, `:357-359`,
  `:361-420`, `:385-386`. ✓
- k≤3 ⇒ cumulant = central moment (`:343-345`); k>4 ⇒ `RuntimeWarning`, returns
  centered moment (`:435-440`). ✓ δ-contact terms never estimated (`:33-40`). ✓
- `estimate_kpoint_slices` def `:445`, anchor leg 0, `base_bins` length k−1, the
  `lag=[0]*k / lag[j]=None` loop, pure-voltage zero-`binned_counts` allocation,
  output `(k−1, 2*max_lag_bins+1)` — `:445-483`, `:469-470`, `:472-483`. ✓
- `compute_kpoint_slice_direct` def `:486`, spike-only reduced re-impl, full-array
  factorial mean `falling_factorial_array(binned_counts[pop],m).mean()/dt_bin**m`
  — `:486-575`, `:559-563`. ✓ `Counter` reuse `:547`. ✓
- Field types `dn`/`dv`/`dm` semantics, "OU uses `dv`" — `:109-121`. ✓

### Supporting cast (`models/`)
- `cumulant_direct_numba._kpoint_slice_direct_k3` `@numba.njit` exists — `:13-14`. ✓
- `hawkes_sim_numba` returns `(binned_counts, voltage_bins, total_spikes)` `:113`;
  "MUST be a plain .py file …" quote `:6-10`; "~15M steps/sec on Apple M1" `:37`. ✓
- `hawkes_sim_multipop_numba`: per-pair `F_{ij}` exponential filters, matches
  `theories/multipopulation_test.theory.py` — `:5-6`, `:11`. ✓
- `*_gtas_numba`: fourth return value `ext_binned_counts` (the `'dm'` field) — e.g.
  `hawkes_sim_linear_expg_gtas_numba.py:316`. ✓
- `hawkes_sage.py` exists. ✓ `spatial_field_phi6_1d_sim.py` = cubic+quintic /
  Allen-Cahn+quintic force `:3-4`. ✓

### `spatial_field_1d_sim.py`
- Lattice dispersion `ω = μ + (2D/dx²)(1−cos(2πm/N))` ≡ manual's `μ + (2D/dx²)(1−cos(k·dx))`
  (since k·dx = 2πm/N) — `_dispersion` `:35-43`. ✓
- ETD1 integrating factor `(1−e^{−ωdt})/ω` `:77`; stationary var `T·N²/(L·ω_k)`
  `:78`; per-step OU increment var `(1−e^{−2ωdt})·stat_var` `:79`; `lap_eig`/`ik_eig`
  `:70,75`; real-mode bookkeeping `:83-87`; derivative vertices (Model B `g_lap`,
  Burgers, KPZ) `:98-107`; Hermitian noise `:114-118`. ✓ **All exact.**
- Inside-body Sage-Integer guards `N=int(N); … seed=int(seed)` `:141-145`;
  `np.random.default_rng(seed)` `:154`; `rng.standard_normal(M)` `:114-115`. ✓
- Estimators `structure_factor` `:175`, `equal_time_correlator` `:191`,
  `lattice_sum_variance` `:205`, `third_cumulant_x` `:215`; `simulate` `:126`. ✓

### `coupled_rd_1d_sim.py`
- `simulate_coupled_rd_1d` def `:142`; returns dict
  `{'C','C_err','C_rep','x_grid','taus','meta'}` `:157-158`. ✓ **Exact.**
- Semi-implicit Crank–Nicolson / Padé(1,1) propagator `B^s ~ e^{−Aτ}`, discrete
  stationary covariance = continuous Lyapunov `AΣ + ΣAᵀ = Nnoise` exactly,
  lag bias O(dt²) — `:30-44`. ✓ (Manual's "exactly to O(dt²)" slightly conflates
  the *exact* stationary covariance with the *O(dt²)* lag propagator, but the
  caveats-list phrasing "O(dt²)-exact in the stationary covariance" is faithful;
  not flagged as a finding.)
- Courant assert `dt·max(D)/dx² < 0.4` (AssertionError) — `:212-214`. ✓
- Oracle `coupled_box_correlator` def `:306`; `expm(-A*abs(tau))` `:353`;
  `solve_continuous_lyapunov(A, Nnoise)` `:351`; scipy import `:73`. ✓

### Cross-file / result keys / numbers
- Quickstart payoff: theory `C_xx(0)=0.9496`, sim `0.9464 ± 0.0009` — real in both
  the notebook output (`temporal_ou_quartic_white.ipynb:220`) and
  `02-quickstart.tex:574`. Residual 0.0032 ≈ 3.6 SE; manual's "about three standard
  errors" is a fair round. ✓
- Result keys `C_tau` / `C_tau_by_ell` produced in `pipeline/compute.py`
  (`:587,791,808`); `C_tau_slices` in `notebooks/daedalus.py:639`;
  `dd.plot_cumulant` at `notebooks/daedalus.py:725`;
  `res['_resolved']['parameters']` present in the notebook and added per commit
  150d112. ✓
- "No SageMath / nauty / networkx / sympy; scipy.linalg in exactly one place" — both
  primary files import only `numpy`(+`numba`/`Counter`); scipy only in
  `coupled_rd_1d_sim.py`. ✓

### LaTeX structural
- Environments balanced: lstlisting 18/18, gotcha 10/10, equation 5/5, itemize 4/4,
  enumerate 3/3, center 2/2, description 2/2, align 1/1 (6 cols both rows),
  tikzpicture/longtable/note/defn/quote 1/1. ✓
- All macros used (`\msrjd \file \code \term \avg \dd \ee \Dt`) and tcolorboxes
  (`note gotcha defn`) defined in `daedalus_manual.tex`. ✓
- Zero bare `_`/`#` outside math/listings (multiline-aware scan). ✓
- All 6 tikz nodes (theory, pipe, sim, res, est, cmp) defined; all `\draw`
  endpoints resolve. longtable has `\toprule/\midrule/\endhead/\bottomrule`. ✓

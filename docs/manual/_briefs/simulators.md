# Validation: Independent Simulators & Cumulant Estimators

> Subsystem slug: `simulators`
> Primary files:
> - `models/ou_langevin_sim_numba.py` (Euler–Maruyama SDE integrators)
> - `models/cumulant_estimator.py` (k-point connected-cumulant estimator)
> Supporting cast (characterized below): `models/cumulant_direct_numba.py`,
> `models/hawkes_sim_numba.py`, `models/hawkes_sim_multipop_numba.py`,
> `models/spatial_field_1d_sim.py`, `models/spatial_field_2d_sim.py`,
> `models/spatial_field_3d_sim.py`, `models/spatial_field_phi6_1d_sim.py`,
> `models/coupled_rd_1d_sim.py`, `models/dendritic_linear_sim_numba.py`,
> and the `hawkes_*_expg*` / `hawkes_*_gtas*` family.

---

## 1. Overview

### What this subsystem is, in plain language

The Daedalus *pipeline* takes a field-theory action (the MSR-JD action of
some stochastic dynamical system), enumerates Feynman diagrams, assigns
propagators and vertices, and **analytically/semi-analytically computes
the connected correlation functions (cumulants)** of the fields — order
by order in a loop expansion. That is the "automated Feynman
calculation" half of the repository.

This subsystem is the **other half: the ground truth.** It contains a
set of *completely independent* numerical simulators that solve the *same*
stochastic equations of motion directly — by brute-force time-stepping —
and a set of **estimators** that turn the raw simulated time-series into
empirical estimates of the very same cumulant slices the pipeline
predicts. The two are then plotted on top of each other. If the
diagrammatic machinery is correct, the curves overlap (within Monte-Carlo
error bars). This is the **trust chain**: a closed loop where an
analytic prediction is checked against a numerical experiment that shares
*no code* with the prediction.

The simulators are deliberately **dumb and direct**:

- **Langevin / SDE simulators** (`ou_langevin_sim_numba.py`,
  `spatial_field_*_sim.py`, `coupled_rd_1d_sim.py`) integrate a
  stochastic *differential* equation `dx/dt = drift(x) + noise` by the
  Euler–Maruyama scheme (or an exponential-Euler / Crank–Nicolson
  variant for stiff linear parts). No diagrams, no enumeration — just a
  `for` loop adding `dt·drift + √(2D·dt)·N(0,1)` each step.
- **Hawkes / point-process simulators** (`hawkes_sim_*numba.py`) integrate
  a self-exciting Poisson point process: each timestep draws
  `Poisson(λ_i·dt)` spikes and feeds them back into the rate. This is a
  *Gillespie-flavoured* (here, fixed-step Poisson-draw) realization, not
  an SDE.
- **Cumulant estimators** (`cumulant_estimator.py`,
  `cumulant_direct_numba.py`) take the binned output of either family and
  estimate the connected k-point cumulant density as a function of time
  lag, with the discrete-spike "shot-noise" diagonal removed.

### Where it sits in the end-to-end pipeline

```
  theory file  ──►  PIPELINE (enumerate → assign → integrate)  ──►  res['C_tau'], res['C_tau_slices'], …
 (action S)                                                              │  analytic cumulants
                                                                        ▼
                                                              ┌──────────────────────┐
                                                              │  sim-compare notebook │  ← plots overlay
                                                              └──────────────────────┘
                                                                        ▲
  theory file  ──►  SIMULATOR (Euler–Maruyama / Poisson)  ──►  binned trajectory  ──►  ESTIMATOR  ──► sim cumulant slices
 (same params)         models/*_sim*.py                       (npop, n_bins)        cumulant_estimator.py
```

- **Feeds the simulators:** the *physical parameters* resolved by the
  pipeline. In the sim-compare notebooks the call is literally
  `fp = res['_resolved']['parameters']` followed by
  `mu = float(fp['mu'])`, etc. — the simulator is handed the *same*
  numbers the theory used, cast to plain Python floats. The simulators
  do **not** read the action; a human has hand-coded the EOM in each
  `sim_*` function to match a specific theory file (the docstrings name
  the matching `theories/*.theory.py`).
- **Consumes the simulator output:** the estimators. A simulator returns
  a binned trajectory array of shape `(npop, n_bins)`; the estimator
  cross-correlates that to produce a cumulant slice `(n_tau,)` or stack
  `(k-1, n_tau)`.
- **Consumes the estimator output:** the comparison notebooks
  (`notebooks/temporal/pipeline_*_sim_compare.ipynb`,
  `notebooks/examples/temporal_*.ipynb`) and the spatial notebooks. They
  overlay `sim['C']` against `res['C_tau_by_ell']` / `res['C_tau']`.

> **Critical framing (do not lose this):** *These files are NOT part of
> the pipeline.* They never run during a normal cumulant computation.
> They exist solely so a skeptical reader can confirm the diagrams give
> the right answer. The repository's confidence in every analytic result
> rests on these brute-force checks agreeing.

---

## 2. The math

### 2.1 The Langevin SDE and Euler–Maruyama

The scalar simulators integrate an Itô stochastic differential equation

```
    dx/dt = f(x) + √(2D)·η(t),      ⟨η(t) η(t')⟩ = δ(t − t')
```

where `f(x)` is the deterministic **drift** and `η` is unit-strength
Gaussian white noise. For the quartic double-well (the flagship
example, `sim_ou_quartic_numba`)

```
    f(x) = −μ·x − ε·x³,
```

so the EOM is `dx/dt = −μx − εx³ + √(2D)η`. This corresponds to the
MSR-JD action quoted in the docstring (`ou_langevin_sim_numba.py:29`):

```
    S = ∫ dt  x̃·((∂_t + μ)·x + ε·x³) − D·x̃²,
```

where `x̃` is the MSR-JD *response* (hatted) field. The `−D·x̃²` term is
the noise vertex; its coefficient `D` fixes the noise strength
`⟨ξ ξ⟩ = 2D·δ(τ)` with `ξ = √(2D)·η`.

**Euler–Maruyama** is the stochastic analogue of the forward-Euler ODE
scheme. Over a step `dt` the deterministic part advances by `dt·f(x)`,
and the *integrated* noise increment `∫_t^{t+dt} √(2D)·η dt'` is a
Gaussian with mean 0 and variance `2D·dt` (because `Var[∫η] = ∫∫δ = dt`).
Therefore one step is

```
    x(t+dt) = x(t) + dt·f(x) + √(2D·dt)·N(0,1).
```

Note the **`√dt` scaling of the noise** (not `dt`): this is the defining
feature of Euler–Maruyama versus a deterministic Euler step. In code
(`ou_langevin_sim_numba.py:74,80–82`):

```python
sqrt_2D_dt = np.sqrt(2.0 * D * dt_sim)
...
x = (x
     + dt_sim * (-mu * x - eps * x * x * x)
     + sqrt_2D_dt * np.random.randn())
```

`np.random.randn()` draws one `N(0,1)`. Euler–Maruyama has weak order
1.0; the bias in the *stationary* statistics scales as `O(dt)`, hence the
docstring insistence that `dt_sim ≪ 1/μ` (the relaxation time).

**Stationary reference (sanity check built into the docstring).** For
the *pure* OU process (`ε = 0`, `f = −μx`) the stationary distribution is
Gaussian with variance `D/μ`. So `C_xx(0) = ⟨x²⟩ = D/μ` is the analytic
target at tree level; finite `ε` shifts it by `O(ε·D²/μ³)`
(`ou_langevin_sim_numba.py:64–66`). This is exactly the number the
pipeline's loop expansion should reproduce.

**Sign conventions / bifurcation structure** (`:60–63`): positive `μ` =
stable origin (sub-critical), `μ = 0` = pitchfork bifurcation, `μ < 0` =
double well with minima at `x = ±√(|μ|/ε)`. The equilibrium Boltzmann
potential is `U(x) = (μ/2)x² + (ε/4)x⁴` (and `+ (γ/6)x⁶` for the sextic
variant), bounded below — hence stable/normalizable — when all
coefficients are positive.

### 2.2 Colored noise via an auxiliary OU process

Some theories use *temporally correlated* (colored, Lorentzian) noise
rather than white noise. `sim_ou_quartic_colored_numba` realizes this by
running a second SDE — an **auxiliary Ornstein–Uhlenbeck process** `ξ` —
and using *its* output as the forcing of `x`:

```
    dξ/dt = −ξ/τc + (√(2D)/τc)·η(t)      →   ⟨ξ(t)ξ(t')⟩ = (2D/τc)·e^{−|t−t'|/τc}
    dx/dt = −μx − εx³ + ξ(t).
```

The key trick: the auxiliary OU process is **discretized EXACTLY**, not
by Euler–Maruyama. A linear OU SDE has the closed-form one-step update

```
    ξ(t+dt) = decay·ξ(t) + σ·N(0,1),
    decay = e^{−dt/τc},     σ² = (2D/τc)(1 − decay²).
```

This preserves the stationary variance for **any** `dt/τc` ratio — there
is no `O(dt)` bias in the noise statistics, only in the `x` Euler step
(`ou_langevin_sim_numba.py:144–148`):

```python
decay = np.exp(-dt_sim / tauc)
var_factor = 1.0 - decay * decay
sigma_xi = np.sqrt((2.0 * D / tauc) * var_factor)
```

> **Subtle factor-of-2 trap** (`:127–132`): in the *white limit*
> `τc → 0`, this colored driver tends to `⟨ξξ⟩ → 4D·δ(τ)` (note **4D**,
> not 2D), because of the `2D/τc` coefficient convention. To compare the
> colored sim against the white sim you must use `D_white = 2·D_colored`.

### 2.3 Cross-correlated multi-field noise (Cholesky)

The 2-field correlated simulators (`sim_ou_quartic_two_dim_corr_numba`,
`sim_ou_quartic_two_dim_color_corr_numba`) drive two fields with jointly
Gaussian white noise of correlation `ρ`:

```
    ⟨η_α(t) η_β(t')⟩ = C_αβ·δ(t−t'),     C = [[1, ρ], [ρ, 1]].
```

Correlated Gaussians are produced by **Cholesky factorization** of the
covariance matrix `C`. For the 2×2 case the Cholesky factor is
`[[1, 0], [ρ, √(1−ρ²)]]`, so (`:298–303`):

```
    u, v  ←  independent N(0,1)
    η_x = u
    η_y = ρ·u + √(1−ρ²)·v.
```

`ρ ∈ [−1,1]` strictly; outside that `C` loses positive-semidefiniteness
and no real Cholesky factor exists. The code defensively clamps
`1−ρ²` to `≥ 0` (`:332–335`) so an out-of-range `ρ` yields NaN rather
than a crash.

### 2.4 Spatial fields: spectral exponential-Euler (ETD1)

The spatial simulators (`spatial_field_1d_sim.py` and friends) integrate
a *field* SDE on a periodic ring/torus:

```
    ∂_t φ(x,t) = −μφ + D∂_x²φ − λφ³ + η(x,t),
    ⟨η(x,t) η(x',t')⟩ = 2T·δ(x−x')·δ(t−t').
```

A naive Euler–Maruyama in real space is unstable/biased because the
diffusion operator `D∂_x²` makes the system **stiff** (the largest
eigenvalue grows like `D·k_max²`). Instead they use **ETD1**
(exponential time-differencing, order 1), working **in Fourier space**:

- Each Fourier mode `φ̂_k` obeys a *linear* OU SDE with rate
  `ω_k = μ + D·k²` (lattice version `ω_k = μ + (2D/dx²)(1−cos(k·dx))`),
  which is integrated **exactly** (same OU closed form as §2.2) — so the
  `λ=0` stationary spectrum is *unbiased in dt*.
- The nonlinear forcing `−λφ³` is added through the ETD1 integrating
  factor `(1 − e^{−ω dt})/ω` (`spatial_field_1d_sim.py:77`).

The per-mode stationary variance is `⟨|φ̂_k|²⟩ = T·N²/(L·ω_k)`
(`:78`), and the per-step OU increment has variance `(1 − e^{−2ω dt})`
times that (`:79`). Real-mode vs interior-complex-mode bookkeeping
(`:83–87`, `:114–118`) handles the Hermitian structure of the rfft.
Derivative vertices (Model B's conserved `g_lap·∂_x²(φ²)`, Burgers
`−(λ/2)∂_x(φ²)`, KPZ `+(λ/2)(∂_xφ)²`) enter via spectral multipliers
`lap_eig` and `ik_eig` (`:70–75`, `:98–107`).

### 2.5 The Hawkes point process

`sim_hawkes_numba` integrates a linear self-exciting Poisson process:

```
    v_i(t+dt) = v_i + (dt/τ)(−v_i + E_i) + (1/τ)·Σ_j W_ij·spikes_j
    λ_i = max(v_i, 0)·dt
    spikes_i ~ Poisson(λ_i).
```

Each step draws actual integer spike counts and feeds them back into the
voltage. This is the discrete-event counterpart of the Langevin sims and
is what makes the **factorial cumulant** machinery (next section)
necessary.

### 2.6 The factorial cumulant density (the estimator's math)

This is the heart of `cumulant_estimator.py`. For point-process
(spike-train) data, the raw k-point correlation contains **delta-function
contact terms** (a spike is perfectly correlated with itself). Concretely,
the same-bin product of counts `n²` overcounts: it includes "a spike
paired with itself", which is a shot-noise term scaling like `1/dt_bin`
and which is *not* part of the smooth connected correlator the diagrams
compute.

The fix is the **factorial moment**. Replace the ordinary power `n^m`
(at `m` coincident same-population fields in the same bin) with the
**falling factorial**

```
    n^{(m)} = n·(n−1)·(n−2)·…·(n−m+1).
```

For a Poisson variable, `⟨n^{(m)}⟩ = λ^m` exactly — the falling factorial
strips off exactly the self-coincidence terms. The resulting estimator
targets the **factorial cumulant density**: the *smooth, off-diagonal*
part of the connected k-point cumulant, with the δ-function diagonals
removed (`cumulant_estimator.py:8–31`). For `k=2` this is just the
ordinary rate covariance (`:14–16`). The reference is Daley &
Vere-Jones, *An Introduction to the Theory of Point Processes*, Ch. 5
(factorial moment measures) (`:55–57`).

**Connected vs raw.** For `k ≤ 3` the connected cumulant equals the
centered moment, so the centered factorial-corrected product *is* the
cumulant — nothing else to subtract. For `k = 4` the connected cumulant
requires subtracting the three disconnected pair products (`:334–433`):

```
    κ_4(X1,X2,X3,X4) = m_4 − m_2(1,2)m_2(3,4)
                            − m_2(1,3)m_2(2,4)
                            − m_2(1,4)m_2(2,3),
```

with `m_n` centered moments. For `k > 4` the partition-based subtraction
is **not implemented** — the estimator emits a `RuntimeWarning` and
returns the centered k-point *moment* instead of the cumulant (`:435–440`).

**Rate conversion.** A spike-counting field's value `n` (a count per bin)
is divided by `dt_bin` to convert to a rate `n/dt_bin` (units Hz). This
is the `denom = dt_bin` in the code. Voltage fields (`'dv'`) are smooth
and use `denom = 1.0` (no rate conversion, units of volts).

**Lag-dependent normalization** (`:46–48`, `:326–332`): the cross-
correlation at sweep lag `L` averages over only the `n_valid − |L|`
overlapping bin pairs, not the full `n_bins`. This is the *unbiased*
finite-window estimator (dividing by `n_overlap`, not a constant).

**Linear (non-circular) cross-correlation via FFT.** The sweep-axis
cross-correlation `C(L) = Σ_t product(t)·sweep(t+L)` is computed with a
**zero-padded FFT** (`:309–320`):

```
    raw = ifft( fft(product) · conj(fft(sweep)) )
```

The FFT must be zero-padded to `n_fft ≥ 2·n_valid` so the circular
convolution the FFT computes coincides with the desired *linear*
correlation over the valid window (no periodic wraparound contamination).
The lag indexing comment (`:316–319`) is load-bearing:

```python
# ifft(F_product × conj(F_sweep))[L] = Σ_t product(t) × sweep(t − L)
# We want C(+lag) = Σ_t product(t) × sweep(t + lag) … Access as raw[(-lag) % n_fft].
# Do NOT use [::-1] — that shifts the index by 1 bin.
```

---

## 3. External tools used

The two primary files are deliberately **lean**: they touch only `numpy`,
`numba`, and the stdlib `collections.Counter`. The wider `models/`
directory also uses `scipy`. Here is each, from scratch.

### 3.1 NumPy (`import numpy as np`)

**What it is.** NumPy is the foundational numerical-array library for
Python. Its central object is the `ndarray`: a contiguous block of
fixed-type numbers (here, 64-bit floats / ints) with a shape, supporting
vectorized arithmetic, FFTs, linear algebra, and random sampling, all in
compiled C under the hood.

**How this code uses it:**

- **Array allocation / accumulation.** Output buffers:
  `x_bins = np.zeros((1, n_bins))` (`ou_langevin_sim_numba.py:70`),
  `bins = np.zeros((2, n_bins))` for 2-field sims.
- **Random sampling (the noise).** `np.random.seed(seed)`
  (`:68`) seeds the global generator; `np.random.randn()` (`:82`) draws
  a single `N(0,1)`; `np.random.poisson(lam)`
  (`hawkes_sim_numba.py:91`) draws spike counts. **NB:** the *spatial*
  sims use the modern `np.random.default_rng(seed)` generator object
  instead (`spatial_field_1d_sim.py:154`) — `rng.standard_normal(M)` —
  because they are *not* numba-jitted and can use the better API.
- **Math primitives inside the hot loop.** `np.sqrt`, `np.exp`, `np.cos`.
- **FFT (estimator).** `np.fft.fft`, `np.fft.ifft`, `np.conj`
  (`cumulant_estimator.py:313–320`) do the cross-correlation;
  `np.fft.rfft` / `irfft` / `rfftfreq` / `fftfreq` drive the spatial
  spectral integrator and structure-factor estimators.
- **Elementwise factorial.** `falling_factorial_array` builds
  `n·(n−1)·…` with `np.ones_like(n_arr, dtype=float)` and an in-place
  `result *= (n_arr − j)` loop (`cumulant_estimator.py:78–81`).
- **Statistics.** `.mean()`, `.std(ddof=1)` for run-averaging and error
  bars in the notebooks/spatial estimators.

### 3.2 Numba (`import numba` / `@numba.njit`)

**What it is.** Numba is a **just-in-time (JIT) compiler** for a subset
of Python+NumPy. The decorator `@numba.njit` ("no-python JIT") compiles
the decorated function to native machine code via LLVM the *first* time
it is called, then caches that compiled version. The `njit` (a.k.a.
`nopython=True`) mode forbids any fallback to the slow Python interpreter
— if numba can't type-infer everything, it errors rather than silently
running slow. This buys roughly **~100× speedup** over the pure-Python
loop (`hawkes_sim_numba.py:37`: "~15M steps/sec on Apple M1").

**How this code uses it.** Every Euler-step simulator is one
`@numba.njit` function with a plain scalar/array signature:

```python
@numba.njit
def sim_ou_quartic_numba(n_steps, dt_sim, mu, eps, D, x_init,
                         bin_size_steps, n_bins, seed):
```
(`ou_langevin_sim_numba.py:17–21`). The inner time-stepping `for`
loop — millions of iterations of `x = x + dt·drift + √dt·noise` — is
exactly the workload numba excels at: tight scalar arithmetic with no
Python object overhead.

**The Sage-Integer gotcha (the load-bearing reason these are `.py`
files).** This is the single most important tooling fact in the
subsystem. When a notebook runs under the **SageMath kernel**, Sage's
*preparser* rewrites the source of every notebook cell: an integer
literal `0` becomes `Integer(0)` (a SageMath ring element), and `1.5`
becomes `RealNumber('1.5')`. These are *not* Python `int`/`float`; they
are objects from Sage's algebra system. Numba's type inference **cannot
handle** Sage ring elements — it only knows native machine types
(`hawkes_sim_numba.py:5–11`):

> This file MUST be a plain `.py` file (not executed through SageMath's
> preparser) because SageMath's preparser converts integer literals like
> `0` to `Integer(0)` … which Numba's type inference cannot handle.

Two consequences enforced throughout:

1. **The simulators live in `.py` modules, never in notebook cells.**
   Only *cell source* is preparsed; imported `.py` files are not. So
   `from models.hawkes_sim_numba import sim_hawkes_numba` is safe.
2. **Every argument at the call site must be cast to a plain Python
   `int`/`float`** before the njit function sees it
   (`hawkes_sim_numba.py:22–24`, and the notebook line
   `mu = float(fp['mu'])`). The non-numba spatial sims defend the same
   way *inside* the function: `N = int(N); n_steps = int(n_steps); …`
   (`spatial_field_1d_sim.py:141–145`) because
   `np.random.default_rng` also rejects Sage `Integer`.

> This is why the brief's focus line says "Sage ring scalars must be cast
> to float": it is not stylistic — an uncast Sage `Integer` will crash
> numba type inference at compile time, or crash `default_rng` at runtime.

### 3.3 `collections.Counter` (`from collections import Counter`)

**What it is.** A stdlib dict subclass that counts hashable items:
`Counter(['a','a','b'])` → `{'a': 2, 'b': 1}`.

**How this code uses it.** In the estimator, when several external legs
land in the *same bin*, they are grouped by `(population, field_type)`
and counted to get the *multiplicity* `m` — the number of coincident same
fields, which is exactly the order of the falling factorial to apply
(`cumulant_estimator.py:279–282`):

```python
pop_ft_multiplicities = Counter(
    (pop_indices[i], field_types[i]) for i in leg_indices)
for (pop, ft), m in pop_ft_multiplicities.items():
    ...
```
and analogously in `compute_kpoint_slice_direct` (`:547`). `m` then
selects between linear centering (`m == 1`) and factorial correction
(`m ≥ 2`).

### 3.4 SciPy (`from scipy.linalg import …`) — coupled-RD oracle only

**What it is.** SciPy is the scientific-computing layer built on NumPy;
`scipy.linalg` is its dense linear-algebra module (LAPACK wrappers).

**How this code uses it** (only in `coupled_rd_1d_sim.py:73`):

- `scipy.linalg.expm(A)` — the **matrix exponential** `e^A`. Used per
  Fourier mode to build the exact lag propagator
  `G(τ) = e^{−A(q)·|τ|}` (`coupled_rd_1d_sim.py:353`).
- `scipy.linalg.solve_continuous_lyapunov(A, Q)` — solves the
  **continuous Lyapunov equation** `A·Σ + Σ·Aᵀ = Q` for `Σ`. This `Σ` is
  the exact stationary covariance of the linear coupled system at each
  mode (`:351`). Together these give `coupled_box_correlator`, the
  *analytic* oracle (no simulation) the coupled Dyson series is validated
  against.

### 3.5 Tools this subsystem does NOT use

For the manual reader's calibration: unlike the pipeline, this subsystem
uses **no SageMath** (only avoids it), **no nauty/networkx** (no graph
enumeration — there are no diagrams here), and **no sympy** (no symbolic
algebra — the EOMs are hand-coded as numeric arithmetic). That cleanness
is the point: the validators share no symbolic machinery with the code
they validate.

---

## 4. Components

### 4.1 `models/ou_langevin_sim_numba.py`

All five functions are `@numba.njit` and share the same skeleton:
seed → init state → loop `n_steps` doing an Euler step + bin accumulation →
emit bin-averaged trajectory. The bin accumulator averages `x` over
`bin_size_steps` consecutive Euler steps to produce one output bin; the
`if cur_bin < n_bins` guard stops writing once the output buffer is full
(extra Euler steps beyond `bin_size_steps × n_bins` are simply ignored).
Output shape always matches the Hawkes `voltage_bins` convention
`(npop, n_bins)` so the estimator can consume it with `field_types=['dv',…]`.

---

**`sim_ou_quartic_numba`** — `ou_langevin_sim_numba.py:17`

```python
@numba.njit
def sim_ou_quartic_numba(n_steps, dt_sim, mu, eps, D, x_init,
                         bin_size_steps, n_bins, seed)
```
- **Takes:** `n_steps` (int, Euler steps), `dt_sim` (float, step size),
  `mu, eps, D` (floats: restoring, cubic, noise), `x_init` (float,
  initial value — use the MF saddle 0 for stationary stats),
  `bin_size_steps` (int, Euler steps per output bin), `n_bins` (int),
  `seed` (int).
- **Returns:** `x_bins : np.ndarray (1, n_bins)` — bin-averaged `x`.
- **Steps:** (1) `np.random.seed(seed)`; (2) precompute
  `sqrt_2D_dt = √(2D·dt_sim)`; (3) loop `n_steps`: accumulate `x` into
  `accum` if the current bin is unfilled, then Euler-step
  `x += dt·(−μx − εx³) + sqrt_2D_dt·randn()`; (4) every `bin_size_steps`
  steps flush `accum/bin_size_steps` into `x_bins[0, cur_bin]`, reset,
  advance bin. Matches `theories/ou_quartic_double_well.theory.py`.

**`sim_ou_quartic_colored_numba`** — `ou_langevin_sim_numba.py:93`

```python
@numba.njit
def sim_ou_quartic_colored_numba(n_steps, dt_sim, mu, eps, D, tauc,
                                 x_init, bin_size_steps, n_bins, seed)
```
- **Takes:** as above plus `tauc` (float, noise correlation time).
- **Returns:** `x_bins (1, n_bins)`.
- **Steps:** runs *two* coupled processes. The auxiliary OU noise `ξ` is
  updated with the **exact** OU discretization
  `ξ_new = decay·ξ + sigma_xi·randn()` (`decay = e^{−dt/τc}`,
  `sigma_xi = √((2D/τc)(1−decay²))`, `:146–148`); then `x` is Euler-
  stepped with `ξ` (not `ξ_new`) as the forcing:
  `x_new = x + dt·(−μx−εx³ + ξ)` (`:162–163`). Note it uses the
  *previous-step* `ξ`, then commits both. Matches the κ² block of
  `theories/ou_quartic_colored.theory.py`.

**`sim_ou_sextic_numba`** — `ou_langevin_sim_numba.py:178`

```python
@numba.njit
def sim_ou_sextic_numba(n_steps, dt_sim, mu, eps, gamma, D,
                        x_init, bin_size_steps, n_bins, seed)
```
- **Takes:** as `sim_ou_quartic_numba` plus `gamma` (float, quintic
  coefficient).
- **Returns:** `x_bins (1, n_bins)`.
- **Steps:** identical skeleton; drift is `−μx − εx³ − γx⁵` (computed as
  `-mu*x - eps*x2*x - gamma*x2*x2*x` with `x2 = x*x`, `:227–229`).
  Confining sextic potential `U = (μ/2)x² + (ε/4)x⁴ + (γ/6)x⁶` is bounded
  below for positive coefficients. Matches `theories/ou_sextic.theory.py`.

**`sim_ou_quartic_two_dim_color_corr_numba`** —
`ou_langevin_sim_numba.py:241`

```python
@numba.njit
def sim_ou_quartic_two_dim_color_corr_numba(
        n_steps, dt_sim, mu1, mu2, eps1, eps2, J1, J2, D1, D2,
        tauc, rho, x_init, y_init, bin_size_steps, n_bins, seed)
```
- **Takes:** two-field params `mu1,mu2,eps1,eps2` (per-field linear/cubic),
  `J1,J2` (cross-coupling: `J1·y` enters `x`'s drift, `J2·x` enters
  `y`'s), `D1,D2` (noise strengths), `tauc` (color time), `rho` (noise
  cross-correlation), `x_init, y_init`.
- **Returns:** `bins : np.ndarray (2, n_bins)` — `bins[0]=x, bins[1]=y`.
- **Steps:** exact OU update for *both* auxiliary noises `ξ_x, ξ_y`, with
  the **white drivers** `(η_x, η_y)` Cholesky-correlated
  `η_x=u, η_y=ρu+√(1−ρ²)v` (`:351–352`); then Euler-step both fields with
  the colored forcing and cross-coupling drift (`:359–362`). Matches
  `theories/ou_quartic_two_dim_color_corr.theory.py`. White-limit check:
  `τc→0` gives `⟨ξ_α ξ_β⟩ → 2D_α C_αβ δ(τ)` (`:306–309`).

**`sim_ou_quartic_two_dim_corr_numba`** — `ou_langevin_sim_numba.py:382`

```python
@numba.njit
def sim_ou_quartic_two_dim_corr_numba(
        n_steps, dt_sim, mu1, mu2, eps1, eps2, J1, J2, D1, D2, rho,
        x_init, y_init, bin_size_steps, n_bins, seed)
```
- Same two-field dynamics but **white** cross-correlated noise (no
  `tauc`). Noise increments `sqrt_2D1_dt·η_x`, `sqrt_2D2_dt·η_y` with
  Cholesky-correlated `η` (`:460–468`). Matches
  `theories/ou_quartic_two_dim_corr.theory.py`. **Convention caveat
  flagged in the docstring** (`:412–419`): the theory file as written
  declares `coefficient='D1'` (i.e. `⟨ξξ⟩ = D·δ`, half strength) whereas
  this sim uses standard `⟨ξξ⟩ = 2D·δ`. For a quantitative compare, edit
  the theory to `'2*D1'` or halve the sim's `D` — see open question.

**`sim_ou_quartic_two_dim_numba`** — `ou_langevin_sim_numba.py:486`

```python
@numba.njit
def sim_ou_quartic_two_dim_numba(
        n_steps, dt_sim, mu1, mu2, eps1, eps2, J1, J2, D1, D2,
        x_init, y_init, bin_size_steps, n_bins, seed)
```
- The **independent** (ρ=0) white two-field baseline. Drift
  `−μ₁x − ε₁x³ + J₁y` and symmetric for `y`; independent
  `randn()` per field (`:534–535`). Matches
  `theories/ou_quartic_two_dim.theory.py`. Linear stability at `(0,0)`
  requires `|J1·J2| < μ1·μ2` (`:507–509`).

### 4.2 `models/cumulant_estimator.py`

**`falling_factorial_array`** — `cumulant_estimator.py:63`

```python
def falling_factorial_array(n_arr, m)
```
- **Takes:** `n_arr` (int ndarray of counts), `m` (int ≥ 1).
- **Returns:** float ndarray of `n·(n−1)·…·(n−m+1)` elementwise.
- **Steps:** `result = np.ones_like(n_arr, dtype=float)`; loop
  `for j in range(m): result *= (n_arr − j)`. The Poisson identity
  `⟨n^{(m)}⟩ = λ^m` is why this removes self-coincidences.

**`compute_kpoint_slice`** — `cumulant_estimator.py:84` — **the core
estimator.**

```python
def compute_kpoint_slice(binned_counts, dt_bin, pop_indices, lag_bins,
                         max_lag_bins, n_fft=None, field_types=None,
                         voltage_bins=None, ext_binned_counts=None)
```
- **Takes:**
  - `binned_counts (npop, n_bins)` — raw spike counts.
  - `dt_bin` — effective bin width (pass `dt_bin_eff`, not nominal).
  - `pop_indices` — length-`k` list, population index per external leg.
  - `lag_bins` — length-`k` list; **exactly one entry must be `None`**
    (the *sweep* axis), the rest fixed integer lags.
  - `max_lag_bins` — half-width of the output lag window.
  - `n_fft` — FFT length (default: next power of 2 `≥ 2·n_valid`).
  - `field_types` — length-`k` list of `'dn'`/`'dv'`/`'dm'` (or natural
    names like `'n'`,`'vI'` which get normalized). `'dn'` = cortical
    spike fluctuation (factorial-corrected), `'dv'` = voltage (smooth,
    linear centering, no factorial), `'dm'` = external GTaS rate
    (shot-noise like `'dn'`). Default all `'dn'`.
  - `voltage_bins (npop, n_bins)` — required if any leg is `'dv'`.
  - `ext_binned_counts (npop, n_bins)` — required if any leg is `'dm'`.
- **Returns:** `(tau_grid, C_slice)`, each `(2·max_lag_bins + 1,)`.
  `tau_grid = lag_index × dt_bin`; `C_slice` is the factorial-cumulant
  density slice. Units: `Hz^k` for all-spike legs; mixed for spike+voltage.
- **Steps (read carefully — this is the central algorithm):**
  1. **Validate the sweep axis** (`:139–145`): exactly one `None` in
     `lag_bins`, recorded as `sweep_idx`.
  2. **Normalize field types** (`:148–202`): default to `'dn'`; raise if
     `'dv'`/`'dm'` requested without the matching data array; map natural
     names through `_normalize_ft` (peels a trailing population suffix and
     maps by first letter `n→dn, v→dv, m→dm`, else `dn`). The docstring
     of `_normalize_ft` (`:166–200`) records the **bug this fixed**:
     before it, `'n'` was treated as non-spike, so legs used `denom=1.0`
     instead of `dt_bin`, returning `count²` instead of `rate²` — wrong
     by `1/dt_bin²` per leg (×16 at `dt_bin=0.25`).
  3. **Define helpers** (`:204–221`): `_data_for(leg)` picks
     counts/voltage/ext array by field type; `_is_spike_ft` is True for
     `'dn'`/`'dm'`; `_centering_denom` returns `dt_bin` for spike fields,
     `1.0` for voltage.
  4. **Valid window** (`:225–240`): from the fixed lags (plus an implicit
     reference at 0) compute `valid_start = max(0,−min_lag)`,
     `valid_end = n_bins − max(0,max_lag)`, `n_valid`. Raise if
     `n_valid ≤ 2·max_lag_bins`.
  5. **Means** (`:242–251`): per `(pop, field_type)`, computed *on the
     valid window* for consistency.
  6. **Build the fixed-leg product** (`:261–300`): group fixed legs by
     lag value; within each lag, group by `(pop, ft)` via `Counter`. For
     multiplicity `m==1` (or any voltage): linear centering
     `(n − mean)/denom`. For spike fields with `m≥2`: factorial
     `falling_factorial_array(n_arr, m)/denom^m`, then **centered**
     (`raw − raw.mean()`). Voltage at `m>1` is raised to the power `m`
     (no factorial). Multiply all factors into `product`.
  7. **Build the sweep series** (`:302–307`): center the swept field
     `(sweep_arr − sweep_mean)/sweep_denom`.
  8. **Cross-correlate by zero-padded FFT** (`:309–320`):
     `raw_xcorr = ifft(fft(product)·conj(fft(sweep))).real`. Mind the
     lag-index comment (do NOT use `[::-1]`).
  9. **Extract & normalize** (`:322–332`): for each lag in
     `[−max_lag_bins, +max_lag_bins]`, divide
     `raw_xcorr[(-lag) % n_fft]` by `n_overlap = n_valid − |lag|`.
  10. **k=4 cumulant subtraction** (`:346–433`): subtract the three
      disconnected pair products via the nested `_two_point(leg_a, leg_b,
      lag_value)` helper, which is *itself* factorial-corrected when the
      pair is same-pop/same-spike-ft/same-lag and ordinary covariance
      otherwise. **k>4** emits a `RuntimeWarning` (`:435–440`).

**`_normalize_ft`** (nested, `:166`), **`_data_for`** (`:204`),
**`_is_spike_ft`** (`:213`), **`_centering_denom`** (`:218`),
**`_data_for_ft`** (`:254`), **`_two_point`** (`:361`) are the
closures described above; `_two_point` returns the smooth 2-point moment
of two legs at their relative lag, with the same factorial logic.

**`estimate_kpoint_slices`** — `cumulant_estimator.py:445` — **the
notebook-facing convenience wrapper.**

```python
def estimate_kpoint_slices(dt_bin, pop_indices, field_types, base_bins,
                           max_lag_bins, binned_counts=None,
                           voltage_bins=None, ext_binned_counts=None)
```
- **Takes:** `dt_bin`; length-`k` `pop_indices` and `field_types` (leg 0
  is the *anchor*); `base_bins` — a length-`(k−1)` list of integer bin
  offsets (the "base point", one per non-anchor leg); `max_lag_bins`; and
  the data arrays.
- **Returns:** `(tau_grid, C)` with `tau_grid (2·max_lag_bins+1,)` and
  `C (k−1, 2·max_lag_bins+1)` — **row `j−1` is slice `j`** (sweep leg `j`).
- **Steps:** for each `j = 1..k−1`: build `lag` with leg 0 at 0, every
  other non-anchor leg at its `base_bins` offset, and leg `j` set to
  `None` (sweep). Call `compute_kpoint_slice`; stack rows. This mirrors
  the theory side's `res['C_tau_slices']` so the `k−1` sim curves line up
  panel-for-panel with `dd.plot_cumulant`'s `k≥3` slices. Convenience:
  if `binned_counts is None` but `voltage_bins` given, it allocates a
  zero `binned_counts` (pure-voltage legs, `:469–470`).

**`compute_kpoint_slice_direct`** — `cumulant_estimator.py:486` — **the
non-FFT cross-check.**

```python
def compute_kpoint_slice_direct(binned_counts, dt_bin, pop_indices,
                                lag_bins, max_lag_bins)
```
- Same interface/output as `compute_kpoint_slice` but uses an explicit
  double loop over `(sweep_lag, t)` instead of FFTs:
  `O(n_bins × n_lags)` vs `O(n_bins log n_bins)`. Slower, but free of any
  FFT artifact (circular wrap, zero-pad, leakage). Use to cross-check the
  FFT estimator. Note: **this variant only handles spike counts**
  (`'dn'`-style) — no `field_types`, `voltage_bins`, or `ext_binned_counts`
  parameters, and `m≥2` factorial means are approximated by the *full-
  array* mean (`:559–563`), not the valid-window mean.

### 4.3 The rest of `models/` (brief characterization)

| File | Family | What it simulates | Estimator pairing |
|---|---|---|---|
| `cumulant_direct_numba.py` | estimator | `@numba.njit` direct (non-FFT) **k=3** factorial-cumulant slice (`_kpoint_slice_direct_k3`); plain `.py` so numba can compile it | self-contained |
| `hawkes_sim_numba.py` | Hawkes | single-population linear Hawkes; Poisson spikes feed voltage; returns `(binned_counts, voltage_bins, total_spikes)` | `compute_kpoint_slice` field `'dn'`/`'dv'` |
| `hawkes_sim_multipop_numba.py` | Hawkes | multipopulation linear Hawkes with **per-pair exponential synaptic filters** `F_{ij}`; flat E/I neuron index; matches `multipopulation_test.theory.py` | `compute_kpoint_slice` |
| `hawkes_sim_expg_numba.py`, `hawkes_sim_quad_expg_numba.py`, `hawkes_sim_*_gtas_numba.py` | Hawkes | exponential-`g` synaptic-filter and quadratic-rate variants; `gtas` = "GTaS" external spike drive (the `'dm'` field, fourth return value `ext_binned_counts`) | `compute_kpoint_slice` `'dm'` |
| `hawkes_sage.py`, `hawkes_linear_sage.py`, `hawkes_linear_expg.py`, `hawkes_quad_expg.py`, `hawkes_linear_expg_gtas.py`, `hawkes_quad_expg_gtas.py` | Hawkes (Sage) | **Sage-side** theory/orchestration helpers for the Hawkes models (these *are* preparsed); not njit | — |
| `dendritic_linear_sim_numba.py` | Hawkes-like | dendritic linear model sim; imports `compute_kpoint_slice` | `compute_kpoint_slice` |
| `spatial_field_1d_sim.py` | spatial Langevin | 1D scalar φ⁴ on a ring, **spectral ETD1**; `simulate`, `structure_factor`, `equal_time_correlator`, `lattice_sum_variance`, `third_cumulant_x` | its own FFT estimators |
| `spatial_field_phi6_1d_sim.py` | spatial Langevin | 1D φ⁶ (cubic+quintic force); near-copy of the φ⁴ sim | same API |
| `spatial_field_2d_sim.py` | spatial Langevin | 2D scalar on a torus, ETD1; `structure_factor_2d`, `radial_*_2d` | radial FFT estimators |
| `spatial_field_3d_sim.py` | spatial Langevin | 3D scalar, ETD1 | radial estimators |
| `coupled_rd_1d_sim.py` | coupled RD | **N-species** reaction-diffusion with cross-noise + unequal D; semi-implicit Crank–Nicolson (dt-exact stationary cov) or explicit; **+ analytic oracle** `coupled_box_correlator` (uses `scipy.linalg.expm` / `solve_continuous_lyapunov`) | `simulate_coupled_rd_1d` returns `{'C','C_err','C_rep',…}` |

**Spatial estimator functions worth naming** (in `spatial_field_1d_sim.py`):
`structure_factor(snaps, meta)` → `S(q)=⟨|φ_q|²⟩` (`:175`),
`equal_time_correlator(snaps)` → `C(x)` (`:191`),
`lattice_sum_variance(L,N,μ,D,T)` → exact `λ=0` variance reference
(`:205`), `third_cumulant_x(snaps)` →
`κ₃(m)=⟨δφ(x₀)·δφ(x₀+m·dx)²⟩` + stderr (`:215`). These are the spatial
counterparts of `compute_kpoint_slice`: they use the **periodic
translational average** (one circular cross-correlation per snapshot via
FFT) instead of a temporal lag sweep.

---

## 5. Data structures

These are loose tuples/dicts, not formal dataclasses.

### Simulator outputs

- **Temporal Langevin / Hawkes:** a binned trajectory
  `np.ndarray (npop, n_bins)` of `float64`. For the 1-field OU sim
  `npop=1`; for 2-field, `npop=2` (`bins[0]=x`, `bins[1]=y`). Hawkes
  returns a **tuple** `(binned_counts, voltage_bins, total_spikes)`,
  each `(npop, n_bins)` (the last `(npop,)`); `gtas` variants add a
  fourth `ext_binned_counts (npop, n_bins)`.
- **Spatial scalar:** `simulate(...)` → `(snapshots, x_grid, meta)`.
  `snapshots (n_rec, N)` recorded field configs post-burn-in;
  `x_grid (N,)` coordinates; `meta` dict with keys
  `dx, dt, L, N, mu, D, lam, g, g_lap, lam_burg, lam_kpz, T,
  record_every, n_rec, k_max` (the Nyquist UV cutoff `π/dx`).
- **2D spatial:** `simulate_2d(...)` → `(snaps, meta)`; `snaps (n_rec,N,N)`,
  `meta` adds `spatial_dim: 2`.
- **Coupled RD:** `simulate_coupled_rd_1d(...)` → dict
  `{'C', 'C_err', 'C_rep', 'x_grid', 'taus', 'meta'}`. `C` shape
  `(n_tau, N, N[, n_xlag])`; `C_rep` adds a leading replica axis;
  `meta` has `dx, dt, sample_dt, stride, L, n_x, n_fields, n_rec, n_rep,
  scheme, courant, k_max, seed`.

### Estimator inputs/outputs

- **Input:** `binned_counts`/`voltage_bins`/`ext_binned_counts` arrays
  `(npop, n_bins)`; `pop_indices` (len-`k` int list); `lag_bins`
  (len-`k`, one `None`); `field_types` (len-`k` str list).
- **Output:** `(tau_grid, C_slice)` — both `(2·max_lag_bins+1,)`; or from
  `estimate_kpoint_slices`, `(tau_grid, C)` with `C (k−1, n_tau)`.
- **Internal:** `mean_by_pop_ft : dict[(pop:int, ft:str) → mean:float]`;
  `lag_to_legs : dict[lag:int → list[leg_idx:int]]`;
  `pop_ft_multiplicities : Counter[(pop, ft) → m:int]`.

---

## 6. Data flow (concrete)

The canonical wiring, taken verbatim from
`notebooks/examples/temporal_ou_quartic_white.ipynb` (cell 9):

```python
from models.ou_langevin_sim_numba import sim_ou_quartic_numba
from models.cumulant_estimator import estimate_kpoint_slices

fp = res['_resolved']['parameters']          # same physics as the theory run
mu = float(fp['mu']); eps = float(fp['eps']); D = float(fp['D'])   # CAST to float!

dt_sim, dt_bin = 0.01, 0.02
n_steps        = int(T_sim / dt_sim)
bin_size_steps = int(max(round(dt_bin / dt_sim), 1))
dt_bin_eff     = bin_size_steps * dt_sim     # the EFFECTIVE bin width (use this)
n_bins         = int(n_steps // bin_size_steps)
max_lag_bins   = int(tau_max / dt_bin_eff)

k           = int(res['_resolved']['k'])
base_bins   = [int(round(b / dt_bin_eff)) for b in base]   # length k-1
pop_indices = [0]*k
field_types = ['dv']*k                       # OU field x is a 'dv' (smooth)

_ = sim_ou_quartic_numba(1000, dt_sim, mu, eps, D, 0.0,
                         bin_size_steps, 100, 0)            # JIT warmup
for r in range(N_RUNS):
    x_bins = sim_ou_quartic_numba(n_steps, dt_sim, mu, eps, D, 0.0,
                                  bin_size_steps, n_bins, rng_base + r)
    tau_sim, Cj = estimate_kpoint_slices(
        dt_bin_eff, pop_indices, field_types, base_bins,
        max_lag_bins, voltage_bins=x_bins)                 # Cj: (k-1, n_tau)
    C_runs.append(np.asarray(Cj).real)
C_sim = np.array(C_runs).mean(axis=0)
C_err = np.array(C_runs).std(axis=0, ddof=1) / np.sqrt(N_RUNS)
```

Salient points the manual should call out:

1. **The simulator is handed `res['_resolved']['parameters']`** — the
   *pipeline-resolved* physics — so theory and sim share parameters but
   *not* code.
2. **Every scalar is cast** (`float(fp['mu'])`, `int(...)`) before the
   njit call — the Sage-Integer guard in action.
3. **A throwaway warmup call** (`sim_..._numba(1000, ...)`) triggers JIT
   compilation once so it isn't timed in the measurement loop.
4. **The OU field is passed as `voltage_bins` with `field_types=['dv']`**
   — it is a smooth continuous field, so it uses linear centering, *not*
   the factorial spike correction. (Hawkes spike legs would use `'dn'`
   and pass `binned_counts` instead.)
5. **`dt_bin_eff` (= `bin_size_steps × dt_sim`), not the nominal
   `dt_bin`, is what the estimator receives.**
6. **Multiple independent runs** (`N_RUNS`, distinct seeds) give the
   error bar `C_err = std/√N_RUNS`.
7. The comparison: `C_sim` vs `res['C_tau_by_ell'][0]` (tree) and
   `res['C_tau']` (loop-corrected) at the τ=0 index — *that overlap is
   the trust-chain verdict.*

For `k≥3`, `estimate_kpoint_slices` returns one row per swept leg, lining
up panel-for-panel with `dd.plot_cumulant`'s theory slices.

---

## 7. Gotchas & caveats

1. **Sage `Integer`/`RealNumber` will crash numba.** The whole "must be a
   plain `.py` file + cast every call-site scalar with `int()`/`float()`"
   discipline exists for this. Symptom if violated: numba typing error at
   compile time, or `np.random.default_rng` rejecting a Sage `Integer` at
   runtime. (`hawkes_sim_numba.py:5–24`; `spatial_field_1d_sim.py:139–145`.)

2. **Colored-noise factor-of-2.** `sim_ou_quartic_colored_numba`'s white
   limit is `⟨ξξ⟩ → 4D·δ`, not `2D·δ`. To compare against the white sim,
   use `D_white = 2·D_colored`. (`ou_langevin_sim_numba.py:127–132`.)

3. **`sim_ou_quartic_two_dim_corr_numba` vs its theory file disagree on
   the noise coefficient.** The sim uses `⟨ξξ⟩ = 2D·δ`; the theory file
   declares `coefficient='D1'` (= `D·δ`, half strength). Quantitative
   compares need one side fixed (edit theory to `'2*D1'`, or halve the
   sim `D`). Speed tests unaffected. (`:412–419`.) *Recorded as an open
   question — possible latent mismatch.*

4. **`ρ` out of `[−1,1]` yields NaN, not an error.** The Cholesky factor
   clamps `1−ρ²` to `≥0` defensively; an out-of-range correlation
   silently propagates NaN. (`:332–335`, `:445–447`.)

5. **FFT lag indexing is fragile.** The cross-correlation lag is read as
   `raw_xcorr[(-lag) % n_fft]`. The comment explicitly warns **"Do NOT
   use `[::-1]` — that shifts the index by 1 bin"**
   (`cumulant_estimator.py:316–319`). Zero-padding to `n_fft ≥ 2·n_valid`
   is mandatory to avoid circular-wrap contamination.

6. **`field_type` normalization fixed a `1/dt_bin²` bug.** Before
   `_normalize_ft`, natural-name legs like `'n'`/`'nE'` were treated as
   non-spike → `denom=1.0` → returned `count²` instead of `rate²` (×16 at
   `dt_bin=0.25`). Anyone building `field_types = [ef[0] for ef in
   external_fields]` hit it. (`:166–200`.) New code should still prefer
   the canonical `'dn'`/`'dv'`/`'dm'` labels.

7. **k>4 is NOT a cumulant.** For `k>4` the estimator does no
   partition subtraction and returns the centered *moment* with a
   `RuntimeWarning`. (`:435–440`.) The full diagonal/contact δ-terms are
   *never* estimated (they're distributions, not τ-grid functions —
   `:33–40`).

8. **`compute_kpoint_slice_direct` is a reduced re-implementation.** It
   handles spike counts only (no `'dv'`/`'dm'`), and its factorial mean
   uses the *full-array* mean rather than the valid-window mean used by
   the FFT path (`:559–563`). It is a *cross-check*, not a drop-in
   replacement — small discrepancies between the two are expected at the
   window edges.

9. **`compute_kpoint_slice` and `compute_kpoint_slice_direct` duplicate
   logic** (and `cumulant_direct_numba._kpoint_slice_direct_k3` is a
   *third* implementation). They must stay in sync by discipline; there's
   no shared kernel. Flag for the manual: explain *why* three exist (FFT
   speed / non-FFT validation / numba-compiled k=3) and that they are
   intentionally independent.

10. **Euler–Maruyama bias.** All `*_numba` Langevin sims are weak-order-1;
    stationary stats carry `O(dt)` bias, so `dt_sim ≪ 1/μ` is required.
    The colored/spatial sims sidestep this *for the linear part* via the
    exact OU / ETD1 update — but the `x` Euler step (colored) and the
    nonlinear ETD1 splitting (spatial) still carry `O(dt)`/splitting
    error. The coupled-RD `semi_implicit` scheme is `O(dt²)`-exact in the
    stationary covariance by construction (`coupled_rd_1d_sim.py:30–44`).

11. **Coupled-RD Courant guard.** `simulate_coupled_rd_1d` *asserts*
    `dt·max(D)/dx² < 0.4` (`:212–214`) — an `AssertionError`, not a
    graceful message — to keep both schemes stable and the Padé lag bias
    negligible.

12. **"Bin finely" or bias `C_xx(0)` low.** A coarse bin
    (`dt_bin` not `≪ 1/μ`) smooths the field over the bin, so the bin-
    averaged variance underestimates the instantaneous `C_xx(0)`. The
    notebook sets `dt_bin = 0.02` for `μ ~ 1`.

13. **Matched-cutoff principle (spatial).** The simulator realizes the
    *lattice* dispersion `ω = μ + (2D/dx²)(1−cos)`, not the continuum
    `μ + Dk²`. Quantitative validation must compare against a theory
    computed at the *same* `k_max = π/dx`, or against the lattice oracle
    (`lattice_sum_variance`, `coupled_box_correlator(dispersion='lattice')`).
    They differ by `O(dx)` (~4% on `C(0,0)` at `L=20, n_x=128`).

---

## 8. Glossary

- **MSR-JD action** — Martin–Siggia–Rose–Janssen–De Dominicis. The
  field-theory representation of a stochastic dynamical system, with a
  physical field `x` and a conjugate *response* field `x̃` (hatted). The
  drift sits in the `x̃·(…)` term, the noise in the `−D·x̃²` term. The
  simulators integrate the *equation of motion* this action encodes.
- **Euler–Maruyama** — first-order numerical scheme for SDEs:
  `x += dt·drift + √(2D·dt)·N(0,1)`. The `√dt` noise scaling
  distinguishes it from deterministic Euler.
- **Ornstein–Uhlenbeck (OU) process** — the linear SDE `dx = −μx + noise`;
  Gaussian, with stationary variance `D/μ` and exponential
  autocorrelation. Used both as a *theory* (the pure-OU baseline) and as
  the auxiliary *colored-noise generator*.
- **Exact OU discretization** — closed-form one-step update
  `x_new = e^{−μdt}·x + σ·N(0,1)`; unbiased in `dt` for the linear OU SDE.
- **ETD1 (exponential time-differencing, order 1)** — integrator for
  stiff `∂_tφ = Lφ + N(φ)`: treats the linear `L` exactly (per Fourier
  mode) and adds the nonlinear `N` via the factor `(1−e^{−ωdt})/ω`.
- **Crank–Nicolson / semi-implicit** — trapezoidal-rule implicit scheme;
  in `coupled_rd_1d_sim.py` it gives the `O(dt²)`-exact stationary
  covariance via a Padé(1,1) propagator.
- **White / colored noise** — white: `⟨ηη⟩ = δ(τ)` (uncorrelated in
  time). Colored: finite correlation time `τc`, here Lorentzian
  `⟨ξξ⟩ ∝ e^{−|τ|/τc}`, generated by an auxiliary OU process.
- **Cholesky factorization** — `C = LLᵀ` for a symmetric PSD matrix;
  multiplying i.i.d. Gaussians by `L` produces correlated Gaussians with
  covariance `C`. Here the explicit 2×2 form gives `η_y = ρu + √(1−ρ²)v`.
- **Hawkes process** — self-exciting point process: spikes raise the rate
  of future spikes. Simulated by per-step `Poisson(λ·dt)` draws fed back
  into the voltage.
- **Gillespie** — exact stochastic-simulation algorithm for jump
  processes. The Hawkes sims here use a *fixed-step Poisson-draw*
  approximation rather than true Gillespie event scheduling — but the
  brief's framing ("Gillespie/Langevin simulators") covers this jump-vs-
  diffusion distinction.
- **Connected cumulant / k-point cumulant** — the *connected* part of the
  k-point correlation (covariance for k=2, etc.). The quantity the
  pipeline computes and these estimators measure.
- **Factorial moment / falling factorial** — `n^{(m)} = n(n−1)…(n−m+1)`;
  for Poisson `n`, `⟨n^{(m)}⟩ = λ^m`. Replacing `n^m` with `n^{(m)}` at
  coincident bins strips the self-spike shot-noise δ-terms, isolating the
  *smooth* off-diagonal cumulant the diagrams predict.
- **Shot noise / contact / diagonal term** — the `1/dt_bin`-scaling δ-
  function piece of a point-process correlator (a spike correlated with
  itself). Removed by the factorial correction.
- **`'dn'` / `'dv'` / `'dm'`** — estimator field-type labels: spike-train
  fluctuation (factorial-corrected), voltage fluctuation (smooth, linear
  centering), external-drive spike fluctuation (shot-noise like `'dn'`).
- **Sweep axis** — in the estimator, the single leg whose lag is `None`;
  the cumulant is reported as a function of *its* lag, others held fixed.
- **Lag-dependent normalization** — dividing each lag's correlation by the
  number of *overlapping* valid bin pairs (`n_valid − |lag|`) for an
  unbiased finite-window estimate.
- **Structure factor `S(q)`** — equal-time `⟨|φ_q|²⟩`; the momentum-space
  two-point function of a spatial field, `→ T/(μ+Dq²)` at tree level.
- **Matched-cutoff principle** — compare sim and theory at the *same* UV
  cutoff `k_max = π/dx`, since the FD lattice dispersion differs from the
  continuum `q²` at `O(dx)`.
- **Lyapunov equation** — `AΣ + ΣAᵀ = Q`; its solution `Σ` is the
  stationary covariance of the linear SDE `dx = −Ax + noise(Q)`. Solved by
  `scipy.linalg.solve_continuous_lyapunov` in the coupled-RD oracle.
- **numba `@njit` / nopython mode** — JIT-compiles Python to native code,
  forbidding interpreter fallback; ~100× speedup on tight numeric loops.
- **SageMath preparser** — rewrites Sage notebook-cell source so integer
  literals become `Integer`, floats become `RealNumber`; the reason these
  sims must live in (un-preparsed) `.py` files and take pre-cast scalars.
- **Trust chain** — the closed loop: analytic diagram prediction vs
  independent brute-force simulation. Agreement (within error bars) is
  the validation that licenses every analytic result in the repo.

---

## 9. Proposed manual subsections

1. **Why a separate validation layer?** — the trust chain; sims share no
   code with the pipeline; what "agreement within error bars" buys you.
2. **The two simulator families** — Langevin/SDE (Euler–Maruyama) vs
   point-process (Hawkes/Poisson); when each applies.
3. **Euler–Maruyama from scratch** — the `√dt` noise, weak order 1,
   `dt ≪ 1/μ`, the `D/μ` stationary-variance sanity check.
4. **Beyond plain Euler** — exact OU discretization (colored noise),
   spectral ETD1 (spatial stiffness), semi-implicit Crank–Nicolson
   (coupled RD); what bias each removes.
5. **Correlated and multi-field noise** — Cholesky construction; the
   `ρ ∈ [−1,1]` constraint and the NaN guard.
6. **The numba/Sage boundary** — why the sims are `.py` modules, why every
   call-site scalar is cast; the `Integer`-crashes-numba story; JIT warmup.
7. **The cumulant estimator: factorial moments** — shot-noise diagonals,
   falling factorials, the Poisson identity, rate conversion by `dt_bin`.
8. **The estimator algorithm walk-through** — sweep axis, valid window,
   fixed-leg product, zero-padded FFT cross-correlation, lag-dependent
   normalization, the `[::-1]` warning.
9. **k=2, k=3, k=4, and beyond** — moment = cumulant for k≤3; k=4
   disconnected-pair subtraction; the k>4 `RuntimeWarning`; what is *not*
   computed (δ contact terms).
10. **Field types `dn`/`dv`/`dm`** — spike vs voltage vs external; the
    `1/dt_bin²` normalization bug and its fix.
11. **Spatial estimators** — structure factor, equal-time correlator,
    third cumulant via periodic translational averaging; matched-cutoff
    comparison and the lattice oracles.
12. **Coupled multi-field validation** — `simulate_coupled_rd_1d` +
    `coupled_box_correlator` (Lyapunov + matrix exponential) as the
    ground truth for the coupled Dyson series.
13. **Putting it together: a sim-compare notebook** — annotated walk-
    through of `temporal_ou_quartic_white.ipynb` cell 9, from
    `res['_resolved']['parameters']` to the overlay plot.
14. **Caveats checklist** — colored factor-of-2, the `corr` theory-file
    coefficient mismatch, FFT lag indexing, Courant guard, binning bias.

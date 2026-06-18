# Subsystem brief: Spatial Heat-Kernel Propagator & Analytic IFT

**Slug:** `spatial-heatkernel`
**Primary source files:**
- `msrjd/integration/spatial/heat_kernel.py` (483 lines)
- `msrjd/integration/spatial/spatial_correlator.py` (417 lines)
- `msrjd/integration/spatial/temporal_integrate.py` (200 lines — **ORACLE-ONLY**, off the production path)
- `msrjd/integration/spatial/loop_parametric.py` (120 lines — **ORACLE-ONLY**, off the production path)
- supporting: `msrjd/integration/spatial/spatial_reduce.py` (the Symanzik core used by the oracle modules)
- consumer/orchestrator: `msrjd/integration/spatial/pipeline_bridge.py` (where the env knobs and analytic-vs-numerical gating live)
- production analytic-IFT batch core: `msrjd/integration/spatial/full_integrator.py` (`_heat_kernel_x_general`)

---

## 1. Overview

### What this subsystem is, in plain language

This subsystem is the **real-space propagator layer** for spatially-extended MSR-JD
(Martin–Siggia–Rose–Janssen–De Dominicis) field theories. Everything upstream in
Daedalus works in a *mixed* representation: frequency `ω` and a formal Laplacian
operator symbol. The job of this subsystem is to **leave that mixed
representation and land in physical `(t, x)` space** by doing two analytic
transforms in closed form:

1. **Close the frequency (`ω`) contour.** The bare inverse propagator of an
   Allen-Cahn / reaction-diffusion field has the form `A + B·k² + i·ω` (mass +
   diffusion + the standard MSR time-derivative). Inverting and closing the
   `ω`-contour gives a simple exponential decay in time, `θ(t)·e^{−(A+Bk²)t}`.

2. **Inverse-Fourier-transform the spatial momentum `k → x`.** A Gaussian in
   `k` (which is exactly what `e^{−Bk²t}` is) inverse-Fourier-transforms to
   another Gaussian in `x` — the **heat kernel** `(4πBt)^{−d/2}·e^{−x²/4Bt}`.
   This is the central trick: *the inverse FT is done with pencil and paper, not
   with a discrete numerical FT on a `q`-grid.* That is why the subsystem is
   called the "analytic IFT" and why it "retired the q-grid."

The headline closed form (d=1) is

```
G(t, x) = θ(t) · (4π B t)^(−1/2) · exp[ −x²/(4 B t) − A t ]
```

where `A` is the **mass** (the relaxation rate, `μ` for the free theory, `μ +
3λφ*²` at a φ⁴ saddle) and `B` is the **diffusion** (the coefficient of `k²`,
i.e. the `D` in `D·Laplacian`).

### Where it sits in the end-to-end pipeline

Reading the data flow from the propagator-construction side toward the
correlator output:

```
   build_propagator (shared, upstream)
        │  produces K_ft  (the inverse-propagator MATRIX in ω and the
        │  inert Laplacian symbol), plus the action's noise sector
        ▼
   heat_kernel.build_spatial_propagator
        │  reads each diagonal entry of K_ft, substitutes Laplacian → −k²
        │  and (if present) GradX → i·k, then peels off (A, B, V) per field
        │  via extract_mass_diffusion.  Emits the PICKLABLE spatial block
        │  of the prop dict (G_tx_sym, ac_mass, ac_diffusion, ac_drift, …).
        ▼
   heat_kernel.make_g_tx_callables
        │  rebuilds the runtime G_tx callables (closures don't pickle, so they
        │  are reconstructed from G_tx_sym after a fresh build OR a cache load).
        ▼
   spatial_correlator.{free_two_point, compute_spatial_correlator_tree}
        │  TREE-LEVEL correlator C(x,τ): combines the heat kernel with the noise
        │  using the erf-closed-form time integral (heat_kernel.erf_time_integral).
        ▼
   pipeline_bridge.compute_spatial_correlator_via_pipeline (production path)
        │  routes the SAME theory through the shared diagram machinery in (t,k),
        │  certifies the per-mode (μ,D,κ) against the diagram C(q,τ), then does
        │  the analytic q→x FT = Σ_modes free_two_point(...).  At d≥2 it instead
        │  uses radial_inverse_ft / periodic_inverse_ft (numerical radial transform).
        ▼
   full_integrator._heat_kernel_x_general (the LOOP analytic IFT)
        │  batched/vectorized version of gaussian_heat_kernel for the per-chamber
        │  Schwinger samples; this is the production analytic IFT for δC at ℓ≥1.
        ▼
   real-space correlator C(x, τ) returned to compute_cumulants / the notebooks
```

**What feeds it:** the inverse-propagator matrix `K_ft` (from
`build_propagator`), the symbolic `ω`/`Laplacian`/`GradX` symbols, the model
dict (`model['spatial']`, `model['boundary']`, `model['initial']`), and the
numeric parameter map `num_params` (which carries model parameters *and* the
saddle values like `phistar1`, *and* the periodic length `L`).

**What consumes its output:** the tree-level correlator path
(`spatial_correlator`), the production correlator bridge (`pipeline_bridge`),
and — for loop diagrams — the batched analytic IFT in `full_integrator`. The
end consumer is `compute_cumulants(...)`, which the example/demo notebooks call.

### A note on "ORACLE-ONLY" modules

Two of the four primary files — `temporal_integrate.py` and
`loop_parametric.py` — carry a banner at the very top:

> `⚠ ORACLE-ONLY — not on the production path. Superseded by full_integrator.py
> … compute_cumulants does NOT use this module.`

They are kept as **independent numerical cross-checks** (a second, slower way to
get the same number, used in tests to pin the production `full_integrator`). Do
not document them to a reader as "how loops are computed today" — they are *how
loops were validated*. The production loop integrator is `full_integrator.py`.

---

## 2. The math, from the ground up

### 2.1 The MSR-JD inverse propagator for a diffusive field

In the MSR-JD formalism a stochastic PDE is rewritten as a field theory with a
*physical* field `φ` and a *response* (or "tilde") field `φ̃`. For an
Allen-Cahn-like reaction–diffusion equation

```
∂_t φ = D ∇²φ − μ φ − (nonlinear terms) + noise ,    ⟨noise·noise⟩ = 2κ δ
```

the quadratic ("Gaussian", "free") part of the action gives an
**inverse propagator** that, after Fourier transform in time (`∂_t → −iω`, but
the code's convention is `+iω`) and replacing the Laplacian operator by its
momentum symbol, reads

```
K(ω, k) = A + B·k² + i·ω           (the diagonal, single-field case)
```

Here:
- `A` = **mass** = the `k⁰`, `ω⁰` term = relaxation rate. Free theory: `A = μ`.
  At a φ⁴ saddle: `A = μ + 3λφ*²` (the curvature of the potential at the saddle).
- `B` = **diffusion** = the coefficient of `k²`. This is the `D` in
  `D·Laplacian`. Physically positive (anti-diffusion is ill-posed).
- `i·ω` = the standard MSR first-order-in-time derivative, **unit-normalized**
  (its coefficient must be exactly `i`; the code checks this and rejects
  anything else as "not a standard MSR kernel").

The **propagator** is the inverse:

```
G(ω, k) = 1 / (A + B·k² + i·ω)
```

### 2.2 Closing the ω-contour: exponential decay in time

`G(ω,k)` has a single simple pole in the complex-`ω` plane at `ω = i(A + Bk²)`.
The retarded inverse FT in time (closing the contour appropriately) collapses to

```
G_R(k, t) = θ(t) · e^{−(A + B·k²) t}
```

`θ(t)` is the Heaviside step (causality — the retarded propagator is zero for
`t < 0`). Define the **k-dependent mass** `m(k) = A + B·k²` (a.k.a. `μ + Dk²`).

### 2.3 Inverse-Fourier-transforming k → x: the heat kernel

The spatial inverse FT of `e^{−Bk²t}` is the classic Gaussian-integral identity

```
∫ dᵈk/(2π)ᵈ  e^{i k·x} e^{−B k² t}  =  (4π B t)^{−d/2} · e^{−|x|²/(4 B t)}
```

This is the **heat kernel** (the Green's function of the diffusion equation).
Multiplying back the `k`-independent decay `e^{−At}` gives the full real-space
retarded propagator:

```
G_R(t, x) = θ(t) · (4π B t)^{−d/2} · exp[ −|x|²/(4 B t) − A t ]
```

**A and B may be complex** (the code tolerates complex mass/diffusion, e.g. at a
complex saddle or in a Keldysh-rotated combination), in which case the formula
is read as analytic continuation. The d-dependence enters *only* through the
`(4πBt)^{−d/2}` prefactor — the rest is d-agnostic. This is the function
`gaussian_heat_kernel`.

### 2.4 Drift (the k¹ term): advection

A first-derivative term `∂_x φ` (advection / a velocity) appears in the inverse
propagator as a **linear-in-k** term `V·k`. The substitution `∂_x → i·k` makes
it appear; so the inverse propagator generalizes to

```
K(ω, k) = A + V·k + B·k² + i·ω
```

`V` is the **drift** (the `k¹` coefficient). Completing the square in the
exponent, `e^{−(A + Vk + Bk²)t}`, the inverse FT carries an extra factor

```
exp[ −i V x /(2B) + V² t /(4B) ]
```

For `V = i·v` (the physical advection velocity `v`) this **shifts the Gaussian
centre to `x = v·t`** — transport at velocity `v`. Crucially:
- **`V = 0` (Laplacian-only / even kernels) is bit-identical to the pure heat
  kernel** — the drift code path is a no-op.
- For a *gradient nonlinearity* (Burgers `∂_x(φ²)`, KPZ `(∂_xφ)²`) the only
  `∂_x` reaching the bilinear (propagator) sector is the saddle cross-term `∝
  φ*`. At the homogeneous saddle `φ* = 0`, `V → 0`, so the propagator is the
  pure heat kernel even for KPZ/Burgers. Drift is supported for **d=1 only**
  (drift is a vector in d≥2 — out of v1 scope).

### 2.5 Periodic boundary conditions: the image sum

For a field on a ring (periodic box) of period `L`, the periodic Green's
function is the **method of images** — sum the infinite-domain kernel over all
image sources displaced by integer multiples of `L`:

```
G_periodic(t, x) = Σ_{n∈ℤ}  G_inf(t, x + nL)
```

Because the Gaussian tail decays **super-exponentially in n**, the sum truncates
quickly. This is `image_sum`. PBC is **d=1 only** in v1 (higher-d PBC would need
a lattice sum per axis).

### 2.6 The tree-level correlator: noise driving two retarded legs

The two-point correlation function `C(x,τ) = ⟨φ(x₁,t₁) φ(x₂,t₂)⟩` (with `x =
x₁−x₂`, `τ = t₁−t₂`) at tree level is the **noise sourcing two retarded
propagators**:

```
C_ij(x, τ) = 2 D^noise_ij ∫_{−∞}^{min(t1,t2)} dt_v ∫ dx_v
             G^R_i(t1−t_v, x−x_v) G^R_j(t2−t_v, −x_v)
```

For a diagonal heat-kernel propagator the **spatial `x_v` integral collapses by
the heat-kernel semigroup** (convolution of two Gaussians is a Gaussian) and the
remaining `t_v` integral is the erf-closed-form

```
C_ii(x, τ) = (D^noise_i / √(4π B_i)) · ∫_{|τ|}^∞ s^{−1/2} exp(−x²/(4 B_i s) − A_i s) ds
```

The `∫ s^{−1/2} e^{−β/s − α s} ds` is the **erf-split closed form** (this is
`erf_time_integral`). After the substitution `s = w²`, the antiderivative of
`exp(−α w² − β/w²)` is

```
F(w) = (√π / 4√α) [ e^{+2√(αβ)} erf(√α w + √β/w) + e^{−2√(αβ)} erf(√α w − √β/w) ]
```

so the integral equals `2[F(√U) − F(√L)]`. With `α = mass`, `β = x²/4B ≥ 0`.
The semi-infinite case `U → ∞` (valid for `Re √α > 0`) has the simple limit
`F(∞) = (√π/4√α)(e^{2√(αβ)} + e^{−2√(αβ)})`.

There is a useful **static (τ = 0) closed form** that the erf formula must
reproduce: `C(x, 0) = D^noise/(2√(AB)) · e^{−|x|√(A/B)}` in d=1, with the d=2
(modified Bessel `K₀`) and d=3 (Yukawa `e^{−κr}/r`) analogues in
`free_correlator_static_closed_form`.

### 2.7 The loop momentum integral: Symanzik polynomials (oracle path)

For loop diagrams the **momentum integral is done first, analytically** (this
avoids the close-pair precision slow path that time-first integration trips
generically). With a Schwinger parameter `w_e` on each edge, every edge is
`e^{−(μ+Dk_e²)w_e}`. Routing each edge momentum as `k_e = a_e·ℓ + b_e·q` (one
loop momentum `ℓ`, external `q`), the loop integral is a **pure Gaussian**:

```
∫ dᵈℓ/(2π)ᵈ exp(−D Σ_e w_e k_e²)  =  (4πDU)^{−d/2} · exp(−D q² (W − V²/U))
```

with the **Symanzik forms**

```
U = Σ_e a_e² w_e          (first Symanzik polynomial = det Lam at L=1)
V = Σ_e a_e b_e w_e        (the M⁻¹-coupling cross term)
W = Σ_e b_e² w_e
F_reduced = W − V²/U       (the reduced external form = "F/U" per q²)
```

The L-loop generalization replaces these scalars by matrices (`Lam`, `N`, `Q`)
and `U = det Lam`, `Q_eff = Q − Nᵀ Lam⁻¹ N`. This is `spatial_reduce` (C0/C1).
**There are no momentum-dependent poles** — only a smooth Schwinger/time
integral remains, so close-pair singularities can never arise.

### 2.8 The full self-energy assembly (oracle path)

After the momentum integral, the residual **causal-time / Schwinger parameter
integral** is (for a 2-vertex self-energy where all internal edges span the same
inter-vertex time `t`):

```
Σ(q,t) = T^{n_C} ∫_{[|t|,∞)^{n_C}} ∏ ds_C · e^{−μ Σ_e w_e} · I_mom({(a_e,b_e)}, {w_e}, q)
```

where:
- a **retarded** `G_R` edge has fixed duration `w_e = t` (from `θ(t)`);
- a **correlation** `C` edge carries a Schwinger parameter `w_e = s_e`,
  integrated over `s_e ∈ [|t|, ∞)` (since `C(k,Δt) = T∫_{|Δt|}^∞ ds e^{−m_k s}`),
  each contributing a factor `T` (temperature).

This is `sigma_parametric`. Note: the diagram's combinatorial / symmetry factor
`𝒮(Γ)` is applied by the **caller**, not here.

---

## 3. External tools used

This subsystem is unusually light on exotic tooling — it leans on **SageMath**
for the symbolic algebra and **mpmath / numpy / scipy** for the numerics. There
is **no nauty, no numba, and no networkx** in these four files (those live in the
enumeration / type-assignment subsystems). Below, each library is explained from
scratch with the exact import lines and the exact way *this* code uses it.

### 3.1 SageMath (`sage.all`)

**What it is.** SageMath is a large open-source mathematics system built on top
of Python. It bundles dozens of math libraries (Singular, Pari, GMP, Maxima, …)
behind one Python API. For this subsystem the relevant piece is Sage's
**Symbolic Ring** — a computer-algebra engine for manipulating symbolic
expressions (think "Mathematica in Python"). A Sage symbolic expression can hold
unknowns (`mu`, `D`, `phistar1`, `k`, `omega`), be expanded, substituted into,
differentiated, and have polynomial coefficients extracted.

**Exact imports.**

`heat_kernel.py:47`
```python
from sage.all import SR, I as SR_I, matrix
```

`spatial_correlator.py:38` and `pipeline_bridge.py:50`
```python
from sage.all import SR
```

**What each imported name is.**
- `SR` — the **Symbolic Ring** itself. `SR(x)` coerces `x` into a symbolic
  expression. `SR.var('k')` declares a symbolic variable named `k`. Sage
  **caches variables by name**, so `SR.var('GradX')` returns *the same object*
  wherever it is declared — this is load-bearing (the gradient symbol declared in
  the operator-IR lowering and the one declared here are guaranteed identical).
- `I as SR_I` — the symbolic imaginary unit `i` (renamed to `SR_I` to avoid
  clashing with loop counters named `I`/`i`).
- `matrix` — Sage's matrix constructor; `matrix(SR, rows)` builds a matrix whose
  entries live in the symbolic ring.

**Exactly how this code uses Sage.**

1. **Declaring momentum / operator symbols** (`heat_kernel.build_spatial_propagator`):
   ```python
   grad_sym = SR.var('GradX')                  # heat_kernel.py:360
   k_var = SR.var('k', latex_name='k')         # heat_kernel.py:362
   ```

2. **Operator → momentum substitution** (`extract_mass_diffusion`,
   `heat_kernel.py:188-192`):
   ```python
   subs = {lap_sym: -k_var**2}
   if grad_sym is not None:
       subs[grad_sym] = SR_I * k_var           # ∂_x → i·k
   e = SR(kft_entry).subs(subs).expand()
   ```
   `.subs(dict)` substitutes one expression for another; `.expand()` multiplies
   everything out so coefficients can be read.

3. **Checking for residual operator symbols** (`.has`,
   `heat_kernel.py:193`):
   ```python
   if e.has(lap_sym) or (grad_sym is not None and e.has(grad_sym)):
   ```
   `.has(sym)` is `True` iff the expression still contains `sym` — used to reject
   higher-derivative operators (a `Laplacian²` would leave a `k⁴` and a residual
   symbol).

4. **Coefficient extraction** (`.coefficient`, `heat_kernel.py:199-222`):
   ```python
   omega_coeff = e.coefficient(omega, 1)       # the i·ω coefficient
   e0 = e.coefficient(omega, 0)                # the ω-independent part
   B = e0.coefficient(k_var, 2)                # the k² coefficient = diffusion
   V = e0.coefficient(k_var, 1)                # the k¹ coefficient = drift
   A = e0.coefficient(k_var, 0)                # the k⁰ coefficient = mass
   ```
   `expr.coefficient(var, n)` returns the coefficient of `var**n`.

5. **Polynomial degree check** (`.degree`, `heat_kernel.py:210`):
   ```python
   deg = e0.degree(k_var)                       # must be ≤ 2 (no higher derivatives)
   ```

6. **Symbolic simplification for an equality test** (`.simplify_full`,
   `heat_kernel.py:201`):
   ```python
   if (omega_coeff - SR_I).simplify_full() != 0:
   ```
   `.simplify_full()` runs Sage's strongest simplifier so that `(i − i)`
   genuinely collapses to `0`.

7. **Free-symbol introspection + numeric callable** (`_make_numeric`,
   `heat_kernel.py:287-306`):
   ```python
   syms = sorted(expr.variables(), key=str)     # the free symbols, sorted by name
   ...
   val = expr.subs(subs)                         # plug in numbers
   return complex(val)                           # coerce to a Python complex
   ```
   `expr.variables()` returns the free symbols of an expression. The closure it
   builds maps a `**num_params` dict to a Python `complex`.

8. **Building the matrix forms** (`reaction_diffusion_matrices`,
   `heat_kernel.py:284`):
   ```python
   return matrix(SR, Mrows), matrix(SR, Drows), matrix(SR, Vrows)
   ```

9. **Reading the noise sector from the FieldTheory polynomial ring**
   (`spatial_correlator.extract_noise_coefficients`, `:66-90`):
   ```python
   sub = {SR.var(str(kk)): vv for kk, vv in num_params.items()}
   ...
   val = complex(SR(coeff).subs(sub))
   ```
   Here `coeff` is a polynomial-ring coefficient; `SR(coeff)` lifts it into the
   symbolic ring, then `.subs` + `complex()` turns it into a float.

**Why Sage and not sympy here?** The upstream `build_propagator` produces `K_ft`
as a Sage matrix, and the FieldTheory ring is a Sage polynomial ring, so this
layer stays in Sage to avoid a translation round-trip. (Other parts of the
codebase — e.g. `pipeline_bridge._diagram_is_bubble` at `:674` — *do* use sympy;
see §3.6.)

### 3.2 mpmath

**What it is.** `mpmath` is a pure-Python **arbitrary-precision** floating-point
library. Where ordinary Python floats give ~16 significant digits, mpmath can
give 30, 50, 1000 digits on demand. It also supplies high-precision special
functions (`erf`, `sqrt`, `exp`) that work on **complex** arguments. Sage ships
mpmath.

**Exact import** (`heat_kernel.py:46`):
```python
import mpmath as mp
```

**Exactly how this code uses it.** Only in `erf_time_integral`
(`heat_kernel.py:93-129`), the erf-split closed form. The reason mpmath is
needed: the antiderivative involves `e^{+2√(αβ)}·erf(...) + e^{−2√(αβ)}·erf(...)`
— a **near-cancellation of two large exponentials** that loses precision
catastrophically in double precision when `√(αβ)` is large. mpmath computes it at
30 digits and then collapses to a `complex`:

```python
with mp.workdps(dps):                  # set working precision to `dps` decimal digits
    a = mp.sqrt(mp.mpc(alpha))         # complex sqrt at high precision
    b = mp.sqrt(mp.mpc(beta))
    pref = mp.sqrt(mp.pi) / (4 * a)
    ...
    return complex(2 * (F_hi - _F(wL)))
```

Calls used: `mp.workdps(dps)` (a context manager that temporarily sets the
decimal precision), `mp.mpc(...)` (a complex mpmath number), `mp.mpf(...)` (a
real mpmath number), `mp.sqrt`, `mp.erf`, `mp.pi`, `mp.e`.

### 3.3 numpy

**What it is.** numpy is the standard Python array / linear-algebra library:
contiguous typed N-dimensional arrays plus vectorized math, with the heavy
lifting in C/BLAS (releasing the GIL — relevant for the threading gate
elsewhere).

**Exact imports** (`spatial_correlator.py:37`, `temporal_integrate.py:47`,
`loop_parametric.py:36`, `spatial_reduce.py:44`):
```python
import numpy as np
```

**Exactly how this code uses it.**
- **Output correlator arrays.** `compute_spatial_correlator_tree` allocates
  `C = np.zeros((len(tau_grid), len(spatial_grid)), dtype=np.complex128)`
  (`spatial_correlator.py:288`) and fills it pointwise.
- **The radial / periodic inverse FT** (`radial_inverse_ft`,
  `periodic_inverse_ft`): `np.trapz` (trapezoidal numerical integration over the
  q-grid), `np.cos`, `np.sin`, `np.meshgrid` (build the d-dimensional lattice of
  integer mode indices), `np.interp` (linear interpolation of the radial `C(q)`
  onto the discrete lattice magnitudes), `np.where` (zero out modes past the
  cutoff), `np.sum`.
- **The Symanzik core** (`spatial_reduce`): `np.asarray`, einsum-free matrix
  builds via `aw.T @ a` etc., `np.linalg.det(Lam)` (determinant = first
  Symanzik `U`), `np.linalg.solve(Lam, N)` (solve `Lam·X = N` for `Lam⁻¹N`
  without forming the inverse).
- **Gauss–Laguerre quadrature** (`temporal_integrate.sigma_parametric`,
  `:109`): `np.polynomial.laguerre.laggauss(deg)` returns the nodes and weights
  of an `deg`-point Gauss–Laguerre rule, which integrates `∫₀^∞ e^{−x} g(x) dx`
  exactly for polynomials of degree `< 2·deg`. This is used as a tensor rule over
  the correlation-edge Schwinger parameters.

### 3.4 scipy

**What it is.** scipy builds scientific routines on top of numpy: numerical
integration (`scipy.integrate`), special functions (`scipy.special`), etc.

**Exact imports.**
- `temporal_integrate.py:48`: `from scipy import integrate`
- `loop_parametric.py:37`: `from scipy import integrate`
- inside functions: `from scipy.special import j0` (`spatial_correlator.py:315`),
  `from scipy.special import k0` (`spatial_correlator.py:406`).

**Exactly how this code uses it.**
- `integrate.quad(f, lo, hi, **opts)` — adaptive 1-D numerical integration; used
  for the single correlation-edge Schwinger integral in `sigma_parametric` and
  for `sigma_R_kernel`.
- `integrate.dblquad(...)` — adaptive 2-D integration; the 2-correlation-edge
  `sigma_K_kernel` (this is the slow path the Gauss–Laguerre tensor rule
  replaces in `sigma_parametric`).
- `scipy.special.j0` — the Bessel function `J₀`, used in the **d=2 Hankel
  transform** branch of `radial_inverse_ft`.
- `scipy.special.k0` — the modified Bessel function `K₀`, used in the **d=2
  static closed-form** `free_correlator_static_closed_form`.

### 3.5 Python stdlib: `cmath`, `math`, `itertools`, `os`, `typing`

- `cmath` (`heat_kernel.py:42`, `spatial_correlator.py:34`) — complex-valued
  `exp` (`cmath.exp`) for the heat kernel (since `A`, `B` may be complex).
- `math` — real-valued `pi`, `sqrt`, `exp`, `ceil` (used where the argument is
  known real, e.g. the `(4πBt)` prefactor in float mode, the q-grid surface
  factors).
- `itertools.product` (`temporal_integrate.py:107`) — the Cartesian product of
  quadrature-node indices for the `nC ≥ 2` Gauss–Laguerre **tensor** rule.
- `os` (`pipeline_bridge`) — reading the environment-variable knobs (§7).
- `typing.Callable, Optional` (`heat_kernel.py:44`) — type hints only.

### 3.6 sympy (NOT in these files, but in a sibling consumer)

For completeness: `pipeline_bridge._diagram_is_bubble` (`:674`) imports
`import sympy as _sp` to detect a `q·ℓ` cross term in an edge's `k²`. sympy is a
*pure-Python* symbolic algebra library (distinct from Sage's SR). It is not used
by any of the four primary files; mentioning it here only so a reader who greps
the directory understands the two symbolic engines coexist.

---

## 4. Components (exhaustive)

### 4.1 `heat_kernel.py`

#### `class SpatialPropagatorError(Exception)` — `heat_kernel.py:50`
The subsystem's one exception type. Raised whenever the **Tier-1 closed-form
heat-kernel path does not apply**: non-diagonal coupling, a higher-derivative
operator (`k⁴`+), a non-unit `iω` coefficient, negative/zero diffusion, a
missing periodic length, an unsupported dimension, etc. The caller is expected to
catch it and either fall back to a numerical path (Tier 2) or re-raise with a
clearer message.

#### `gaussian_heat_kernel(t, x, A, B, spatial_dim=1, V=0.0)` — `heat_kernel.py:58`
**Signature / inputs.** `t` (scalar time, may be complex), `x` (scalar
separation in d=1; pass `|x|` radial for d≥2), `A` (mass, complex ok), `B`
(diffusion, complex ok), `spatial_dim` (1/2/3), `V` (drift, d=1 only).
**Returns.** A Python `complex` — the value of `G_R(t,x)`.
**Step by step.**
1. Coerce `t` to a real float if it is a non-complex real-like object
   (`heat_kernel.py:74`).
2. **Causality guard:** if `Re(t) ≤ 0` return `0j` (the `θ(t)`). `:75-76`
3. Coerce `A, B, V` to `complex`; `x_signed = float(x)`; `xx = x²`. `:77-81`
4. Build the prefactor `(4π B t)^{−d/2}` with `d = spatial_dim`. `:82`
5. If `V ≠ 0`: require `d == 1` (else raise — drift is a vector in d≥2), and
   add the drift exponent `−i V x/(2B) + V² t/(4B)`. `:84-89`
6. Return `pref · exp(−x²/(4Bt) − A t + drift_exp)` as a `complex`. `:90`

**Caveat.** `V = 0` is bit-identical to the pure heat kernel (the drift branch is
skipped entirely).

#### `erf_time_integral(alpha, beta, L_lo, U_hi=None, dps=30)` — `heat_kernel.py:93`
**Inputs.** `alpha` (mass `α`, complex/real), `beta` (`= x²/4B ≥ 0`, complex/real),
`L_lo` (lower limit, ≥ 0), `U_hi` (upper limit; `None` ⇒ semi-infinite `U → ∞`),
`dps` (mpmath decimal digits, default 30).
**Returns.** Python `complex` — `∫_{L_lo}^{U_hi} s^{−1/2} exp(−β/s − α s) ds`.
**Step by step.**
1. Enter an mpmath precision context `mp.workdps(dps)`. `:110`
2. `a = √α`, `b = √β` (complex sqrt); `pref = √π / (4a)`. `:111-113`
3. Define the antiderivative helper `_F(w)` (the erf-split form), with a careful
   **`w → 0⁺` limit** branch: `erf(±∞) = ±1` when `β > 0`, and `erf(0) = 0` when
   `β = 0`. `:115-122`
4. `wL = √L_lo` (or `0` if `L_lo ≤ 0`). `:124`
5. If `U_hi is None`, use the `U → ∞` limit `F_hi = pref(e^{2ab} + e^{−2ab})`;
   else `F_hi = _F(√U_hi)`. `:125-128`
6. Return `complex(2·(F_hi − _F(wL)))`. `:129`

**Validation note (docstring).** "Verified to machine precision (incl. complex α)
in `docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py`."

#### `image_sum(t, x, A, B, L, spatial_dim=1, eps=1e-12, n_max_cap=2000, V=0.0)` — `heat_kernel.py:132`
**Inputs.** Same `(t, x, A, B, V)` as the kernel, plus `L` (period), `eps`
(relative truncation tolerance), `n_max_cap` (max images per side).
**Returns.** `complex` — the periodic-BC kernel `Σ_n G_inf(t, x + nL)`.
**Step by step.**
1. Require `spatial_dim == 1` (PBC d=1 only) else raise. `:143-145`
2. `base = G(t, x)`; `total = base`; `ref = |base|` (or 1 if zero). `:146-148`
3. Loop `n = 1, 2, …`: add `G(t, x+nL) + G(t, x−nL)`. `:150-153`
4. **Truncate** once both `|term_p|` and `|term_m|` fall below `eps·ref` AND
   `n ≥ 2` (super-exponential Gaussian decay). Cap at `n_max_cap`. `:154-156`
5. Return `complex(total)`. `:157`

#### `extract_mass_diffusion(kft_entry, omega, k_var, lap_sym, grad_sym=None)` — `heat_kernel.py:161`
**Inputs.** `kft_entry` (one diagonal SR entry of the inverse propagator), the
symbols `omega`, `k_var`, `lap_sym` (the inert Laplacian symbol), `grad_sym` (the
inert `∂_x` symbol, optional).
**Returns.** The tuple `(A_expr, B_expr, V_expr)` of SR expressions —
mass / diffusion / drift.
**Step by step.**
1. Substitute `lap_sym → −k_var²` and (if present) `grad_sym → i·k_var`, then
   `.expand()`. `:188-192`
2. **Reject residual operator symbols** (`.has`) — a higher-derivative operator
   leaves one behind. `:193-196`
3. **Check the `iω` coefficient is exactly `i`** (`.coefficient(omega,1)`,
   `.simplify_full()`) — the standard MSR normalization. `:199-204`
4. Take the ω-independent part `e0 = .coefficient(omega, 0)`. `:207`
5. **Check the k-degree ≤ 2** (`.degree(k_var)`). `:209-216`
6. `B = e0.coeff(k², )`, `V = e0.coeff(k¹)`, `A = e0.coeff(k⁰)`. `:217-222`
7. Return `(A, B, V)`. `:223`

#### `reaction_diffusion_matrices(K_ft, omega, k_var, lap_sym, grad_sym=None)` — `heat_kernel.py:226`
**Inputs.** The **full** inverse-propagator matrix `K_ft` (not just one entry),
same symbols.
**Returns.** SR matrices `(M, D, V)` each `N×N` — the **matrix** reaction,
diffusion, drift (the coupled-field generalization of `extract_mass_diffusion`).
**Step by step.** For each entry `K_ft[i,j]`: substitute operators → k; reject
residual symbols; require the `iω` coefficient be `i` on the **diagonal** and
`0` off-diagonal (the time derivative must be diagonal & unit-normalized); reject
k-degree > 2; collect `D[i,j] = coeff(k²)`, `V[i,j] = coeff(k¹)`, `M[i,j] =
coeff(k⁰)`. `:248-284`. **For a diagonal `K_ft` this reduces, per entry, to
`extract_mass_diffusion` bit-identically.** This feeds the coupled-field spectral
propagator (`spectral_propagator.py`, the path the diagonal Tier-1 builder does
*not* handle).

#### `_make_numeric(expr)` — `heat_kernel.py:287`
**Inputs.** An SR expression. **Returns.** A closure `f(**num_params) -> complex`.
Reads the expression's free symbols (`expr.variables()`), and at call time
substitutes each by name from `num_params` (raising `KeyError` if any symbol is
missing), then `complex(val)`. This is how a symbolic mass/diffusion/drift
becomes a numeric value once the parameters and saddle are known.

#### `build_spatial_propagator(K_ft, omega, ns, model, resp_names, phys_names, verbose=True)` — `heat_kernel.py:310`
**Inputs.** `K_ft` (Sage inverse-propagator matrix, rows=response, cols=physical),
`omega` (the `ω` SR var), `ns` (namespace carrying `ns.Laplacian`), `model` (must
have `model['spatial']`; may have `model['boundary']`, `model['initial']`),
`resp_names`/`phys_names` (field-name lists), `verbose`.
**Returns.** A **picklable** dict — the spatial block of the prop dict (see §5).
**Raises.** `SpatialPropagatorError` on the non-Tier-1 case.
**Step by step.**
1. Read `d = model['spatial']['dim']`; require `d ∈ {1,2,3}`. `:336-340`
2. If `d ≠ 1` and the boundary mode is `'periodic'`, raise (PBC is d=1 only).
   `:345-349`
3. Fetch `lap_sym = ns.Laplacian` (raise if absent — not a spatial model).
   `:351-354`
4. Declare `grad_sym = SR.var('GradX')` and `k_var = SR.var('k')`. `:360-362`
5. **Diagonality guard:** require every off-diagonal `K_ft[i,j]` (i≠j) be zero,
   else raise (off-diagonal coupling is a v2 feature). `:369-377`
6. Read the boundary / initial modes and the periodic length name. `:379-387`
7. For each diagonal entry, call `extract_mass_diffusion` to get `(A, B, V)`,
   storing into `ac_mass[i]`, `ac_diffusion[i]`, `ac_drift[i]`, and `G_sym[(i,i)]
   = (A, B)`. Off-diagonal `G_sym` entries are set `None`. `:394-406`
8. If `verbose`, print the per-mode A/B (and drift if nonzero). `:408-414`
9. Return the dict (§5). **The runtime `G_tx` callables are NOT stored** (closures
   don't pickle) — they are rebuilt by `make_g_tx_callables`. `:416-431`

#### `make_g_tx_callables(prop)` — `heat_kernel.py:434`
**Inputs.** A prop dict carrying the picklable spatial block (`G_tx_sym`,
`spatial_dim`, `bc_mode`, `bc_params`, `ac_drift`).
**Returns.** `dict[(i,j)] -> callable (t, x, **num_params) -> complex`, or `None`
if no spatial block. **Idempotent / cheap.**
**Step by step.**
1. Bail with `None` if `G_tx_sym is None`. `:443-445`
2. Read `d`, `bc_mode`, `bc_params`, the periodic length name `L_name`, and
   `ac_drift`. `:446-450`
3. For each `(i,j)` entry: if `None`, install a zero-returning closure;
   otherwise build numeric callables `A_num = _make_numeric(A_expr)`, `B_num`,
   and a drift `V_num` (only if the symbolic drift is nonzero). `:455-466`
4. Build a per-entry closure `_g(t, x, **num_params)` that evaluates `A,B,V`
   numerically and then dispatches:
   - **periodic** → `image_sum(t, x, A, B, L, spatial_dim=d, V=V)` (resolving `L`
     from `num_params` by name, or as an inline number); raises if the length is
     missing;
   - **else** → `gaussian_heat_kernel(t, x, A, B, spatial_dim=d, V=V)`. `:468-480`
5. Return the dict of closures. `:483`

### 4.2 `spatial_correlator.py`

#### `extract_noise_coefficients(ft, num_params)` — `spatial_correlator.py:46`
**Inputs.** `ft` (an expanded FieldTheory; must have `ft._by_tp`), `num_params`.
**Returns.** `{field_index: D_noise}` — the white-noise spectral coefficient per
response field.
**Step by step.**
1. Read the `(2,0)` bigrade sector (`ft._by_tp[(2,0)]` — the `−D^noise·φ̃²` term).
   Raise if the FieldTheory isn't expanded; return `{}` if no noise sector. `:57-62`
2. Build a substitution dict mapping each parameter name to its value. `:65`
3. Iterate the sector polynomial's monomials (`noise_sector.dict().items()`).
   `:74-78`
4. A **pure `φ̃_i²` monomial** has exponent 2 on ring position `i < n_tilde` and
   0 elsewhere; for such a monomial, the white-noise coefficient is `D_noise_i =
   −Re(coeff)` (the action term is `−D_noise φ̃²`). Skip absurdly small values.
   `:80-89`
5. Return the dict. `:90`

#### `extract_noise_matrix(ft, num_params)` — `spatial_correlator.py:93`
**Inputs / returns.** Same `ft`/`num_params`; returns the full `(n_tilde ×
n_tilde)` numpy noise-covariance matrix `N`, `⟨ξξᵀ⟩ = N` — the coupled-field
generalization.
**Convention.** `N_ii = −2·coeff(φ̃_i²)`, `N_ij = −coeff(φ̃_i φ̃_j)` (i≠j). So
`N_ii = 2·κ_i` with `κ_i` the value the diagonal extractor returns. Reads only
the `q⁰` (white-noise) part; `q²`-dependent (conserved / Model-B) noise is
skipped (left 0). `:107-134`

#### `free_two_point(mu, D, kap, x, tau, bc_mode='infinite', L=None, n_images_cap=200)` — `spatial_correlator.py:138`
**Inputs.** `mu` (mass, float/complex), `D` (diffusion > 0), `kap` (noise spectral
coefficient), `x`, `tau`, `bc_mode`, `L` (period), `n_images_cap`.
**Returns.** `complex` — the tree-level diagonal correlator `C(x,τ)`.
**Step by step.**
1. `pref = kap / √(4π D)`. `:150`
2. Define `_C_inf(xx)` = `pref · erf_time_integral(mu, β=xx²/4D, |τ|, U_hi=None)`
   — the infinite-domain erf closed form. `:152-155`
3. `bc_mode == 'infinite'` → return `_C_inf(x)`. `:157-158`
4. `bc_mode == 'periodic'` → sum images `_C_inf(x ± nL)` with the same
   `eps`-style truncation as `image_sum` (here `1e-13·ref`, `n ≥ 2`). Requires
   `L`. `:159-173`
5. Unknown `bc_mode` → raise. `:174`

**Note (argument naming):** the first two arguments are `(mu, D)` = (mass,
diffusion). Callers map their `(A, B)` = (mass, diffusion) onto these — e.g.
`compute_spatial_correlator_tree` calls `free_two_point(A, B, Dn, …)` (`:291`),
which is correct: `A → mu`, `B → D`.

#### `compute_spatial_correlator_tree(ft, model, prop, num_params, external_fields, tau_grid, spatial_grid, verbose=False)` — `spatial_correlator.py:178`
**Inputs.** The FieldTheory, the model, the prop dict, numeric params, the two
external legs (internal names), and the `(τ, x)` grids.
**Returns.** `(C_tau_x, info)`: a complex array of shape `(len(tau_grid),
len(spatial_grid))` and an info dict recording the field index and the actual
`A, B, D_noise, bc_mode, L` used.
**Step by step.**
1. Require a spatial block (`prop['spatial_dim']`). `:192-194`
2. Normalize `num_params` to string keys; resolve the periodic length `L` (by
   name or inline). `:198-211`
3. **Resolve the field index** of the external legs by matching the leg base name
   to the physical column names (stripping trailing population digits). `:215-231`
4. **Tier-1 guard:** if `prop['G_tx_sym'] is None`, raise a clear
   `NotImplementedError` explaining that v1 supports only the diagonal
   heat-kernel propagator (off-diagonal coupling / higher-derivative operators
   are v2). `:239-248`
5. Read `(A_expr, B_expr)` for the diagonal entry; substitute params to get
   numeric `A` (complex) and `B` (real). `:250-254`
6. **Diffusion-sign guard:** `B < 0` → anti-diffusive / ill-posed (raise);
   `B == 0` → no Laplacian / time-only field (raise — a spatial correlator is
   undefined). `:261-275`
7. Read the noise coefficient `Dn` for this field (`extract_noise_coefficients`);
   raise if absent. `:276-281`
8. Fill `C[it, ix] = free_two_point(A, B, Dn, x, τ, bc_mode, L)` over the grid.
   `:288-292`
9. Return `(C, info)`. `:293-295`

#### `radial_inverse_ft(q_grid, Cq, x, spatial_dim)` — `spatial_correlator.py:299`
**Inputs.** `q_grid` (0…k_max), `Cq` (the isotropic momentum-space correlator
`C(|q|)` sampled on the grid; may be complex), `x` (scalar or array),
`spatial_dim` (1/2/3).
**Returns.** `C(|x|)` (matching `x`'s shape; complex if `Cq` is complex).
**The transforms.**
- d=1: `C(x) = (1/π) ∫₀^∞ cos(qx) C(q) dq`
- d=2: `C(x) = (1/2π) ∫₀^∞ q J₀(qx) C(q) dq` (Hankel order 0)
- d=3: `C(x) = (1/2π²x) ∫₀^∞ q sin(qx) C(q) dq` (with the `x→0` limit handled —
  `sin(qx)/x → q`, so the integrand becomes `q²·C(q)`).
Each is a `np.trapz` over the grid. This is the **external output transform**
(distinct from the loop momentum integral): it turns a self-energy-dressed
`δC(|q|,τ)` into real space at d≥2. **It truncates at `k_max = q_grid[−1]` — a
finite cutoff (Regime 1); the continuum value is recovered as `k_max → ∞` with a
fine grid.**

#### `periodic_inverse_ft(q_grid, Cq, x, spatial_dim, L, n_cut=None)` — `spatial_correlator.py:341`
**Inputs.** Same isotropic `Cq` on `q_grid`, plus the box period `L` and an
optional mode cutoff `n_cut`.
**Returns.** The discrete-momentum (lattice-sum) inverse FT on a periodic cubic
box:

```
C(x) = (1/L^d) Σ_{n∈ℤ^d} cos((2π/L) n₁ x) · C(|k_n|),   k_n = (2π/L)·n,  |k_n| ≤ k_max
```

**Step by step.** Build the integer-mode lattice with `np.meshgrid` (d=1/2/3),
compute `|k_n|` per mode, **interpolate** the radial `Cq` onto those magnitudes
(`np.interp`, real and imag separately if complex), zero out modes beyond
`k_max` (`np.where`), and sum `cos(k_x·x)·C(|k_n|)` with prefactor `1/L^d`.
`n_cut` defaults to `ceil(k_max·L/2π)+1`. The continuum limit `L → ∞` recovers
`radial_inverse_ft`. **Both BCs share one momentum-space correlator `Cq`** — only
the inverse transform differs.

#### `free_correlator_static_closed_form(r, mu, D, T, spatial_dim)` — `spatial_correlator.py:398`
**Returns.** The exact static (`τ = 0`) free correlator `C(|r|)` — the closed-form
oracle for `radial_inverse_ft` (the inverse FT of `C(q,0) = T/(μ+Dq²)`):
- d=1: `(T / 2√(μD)) e^{−|r|√(μ/D)}`
- d=2: `(T / 2πD) K₀(|r|√(μ/D))`
- d=3: `(T / 4πD) e^{−|r|√(μ/D)} / |r|`

Uses `scipy.special.k0` for the d=2 Bessel.

### 4.3 `spatial_reduce.py` (the Symanzik core)

#### `symanzik_matrices(a_list, b_list, weights)` — `spatial_reduce.py:58`
Builds the Symanzik matrices `(Lam, N, Q)` from per-edge routing coefficients and
weights. `a` is `(E, L)`, `b` is `(E, n_ext)`, `w` is `(E,)`. `Lam = (a·w)ᵀ·a`
(L×L), `N = (a·w)ᵀ·b` (L×n_ext), `Q = (b·w)ᵀ·b` (n_ext×n_ext). Raises if the row
counts mismatch.

#### `symanzik_polynomials(a_list, b_list, weights)` — `spatial_reduce.py:82`
Returns `(U, Q_eff)`: `U = det(Lam)` (first Symanzik) and `Q_eff = Q − Nᵀ Lam⁻¹
N` (reduced external form). Raises if `U ≤ 0` (no loop-momentum damping). Uses
`np.linalg.det` and `np.linalg.solve`.

#### `momentum_integral(a_list, b_list, weights, q, D, spatial_dim=1)` — `spatial_reduce.py:101`
Evaluates the L-loop Gaussian momentum integral `(4πD)^{−Ld/2} U^{−d/2}
exp(−D qᵀ Q_eff q)`. `q` may be scalar (n_ext=1) or a length-n_ext sequence.
Generalizes `loop_parametric.gaussian_momentum_integral` (the L=1 case) to any
L and d. Returns a float.

### 4.4 `loop_parametric.py` (ORACLE-ONLY)

#### `symanzik_UF(a, b, w, D, spatial_dim=1)` — `loop_parametric.py:41`
The **L=1** Symanzik core. Edges `k_e = a_e ℓ + b_e q`, weight `w_e ≥ 0`.
Computes `U = Σ a²w`, `V = Σ ab·w`, `W = Σ b²w`, `F_reduced = W − V²/U`, and
`prefactor = (4πDU)^{−d/2}`. Returns `(U, F_reduced, prefactor)`. Raises on `U ≤
0`.

#### `gaussian_momentum_integral(a, b, w, q, D, spatial_dim=1)` — `loop_parametric.py:73`
Thin wrapper: `pref · exp(−D q² F_reduced)` — the single-loop Gaussian momentum
integral.

#### `sigma_R_kernel(q, t, mu, D, T)` — `loop_parametric.py:93`
The φ̃φ² 1-loop **response** self-energy bubble kernel `∫dℓ/2π G_R(ℓ,t)
C(q−ℓ,t)` (one response edge `a=1,b=0,w=t`, one correlation edge
`a=−1,b=1,w=s≥t`), via a 1-D `integrate.quad` over `s ∈ [t,∞)`. Returns `T·val`
(the combinatorial/coupling prefactor is applied by the caller).

#### `sigma_K_kernel(q, t, mu, D, T)` — `loop_parametric.py:110`
The **Keldysh** self-energy `∫dℓ/2π C(ℓ,t) C(q−ℓ,t)` (two correlation edges), via
a 2-D `integrate.dblquad` over `(s1, s2) ∈ [|t|,∞)²`. Returns `T²·val`. (This
`dblquad` is the bottleneck the Gauss–Laguerre tensor rule replaces in
`sigma_parametric`.)

### 4.5 `temporal_integrate.py` (ORACLE-ONLY)

#### `sigma_parametric(edges, q, t, mu, D, T, spatial_dim=1, s_cap=None, quad_opts=None)` — `temporal_integrate.py:56`
**Inputs.** `edges` = list of `(a, b, kind)` per internal edge (`a` = loop-coeff
tuple, `b` = external-coeff tuple, `kind ∈ {'R','retarded','C','correlation'}`);
`q` (external momentum); `t` (inter-vertex time); `mu, D, T`; `spatial_dim`;
`s_cap` (upper Schwinger cap); `quad_opts`.
**Returns.** `Σ(q,t)` (float). Includes `T^{n_C}` and `e^{−μΣw}`; the `𝒮(Γ)`
factor is the caller's.
**Step by step.**
1. Split edges into correlation (`c_idx`) and retarded (`r_idx`); validate kinds.
   `:72-78`
2. Set `tt = |t|`, lower limit `lo = max(tt, 1e-9)` (avoid the `U→0` corner),
   upper limit `hi = lo + s_cap` (default `60/μ`). `:80-82`
3. Define `_integrand(svals)`: build the weight vector `w` (retarded edges → `t`,
   correlation edges → the Schwinger sample), call
   `momentum_integral(a_all, b_all, w, q, D, spatial_dim)`, multiply by
   `e^{−μ Σw}`. `:85-92`
4. **Dispatch by number of correlation edges `nC`:**
   - `nC == 0`: evaluate directly. `:94-95`
   - `nC == 1`: adaptive `integrate.quad` over the single Schwinger param. `:96-98`
   - `nC ≥ 2`: a **Gauss–Laguerre tensor rule** (degree 48 for nC=2, else 40)
     over the correlation Schwinger params, with the substitution `s_C = lo +
     x_k/μ` so `∫_lo^∞ e^{−μs} g(s) ds = (e^{−μ lo}/μ) Σ_k w_k g(lo + x_k/μ)`.
     Iterates `itertools.product` over node indices. ~10–100× faster than the
     adaptive dblquad/nquad it replaces. `:99-122`
5. Return `T^{n_C} · val`. `:123`

#### `bubble_edges(kind_R='R')` — `temporal_integrate.py:127`
Convenience edge-spec for the φ̃φ² 1-loop bubble: `[((1,),(0,),kind_R),
((−1,),(1,),'C')]`. `kind_R='R'` → Σ_R (one G_R + one C); `kind_R='C'` → Σ_K
(both correlation).

#### `sunset_edges()` — `temporal_integrate.py:135`
Convenience edge-spec for the 2-loop sunset: three correlation edges
`ℓ₁`, `ℓ₂`, `q−ℓ₁−ℓ₂`.

#### `bubble_delta_equal_time_via_C(q, mu, D, T, g=1.0, C_R=4.0, C_K=2.0, n_a=160, a_max=None, spatial_dim=1)` — `temporal_integrate.py:144`
End-to-end C0→C1→C2→C3-lite: tabulate `Σ_R(q,a)` and `Σ_K(q,a)` via
`sigma_parametric`, then collapse via the MSR Dyson equation

```
δC(q,0) = g²·[ C_R·(T/m²)∫₀^∞ Σ_R(a)e^{−ma}da + C_K·(1/m)∫₀^∞ Σ_K(a)e^{−ma}da ]
```

with `m = μ + Dq²`, `C_R = 4`, `C_K = 2` (d-independent topology constants). At
d=1 reproduces `loop_dyson.bubble_delta_S` (the backend-B golden reference);
d=2,3 give the higher-d bubble. Uses `np.trapz`.

#### `bubble_delta_phi2_via_C(mu, D, T, g=1.0, spatial_dim=1, q_max=None, n_q=80, C_R=4.0, C_K=2.0)` — `temporal_integrate.py:173`
The momentum-integrated equal-time bubble variance correction `δ⟨φ²⟩ =
∫dᵈq/(2π)ᵈ δC(q,0)` = `(S_{d−1}/(2π)ᵈ) ∫₀^∞ q^{d−1} δC(q,0) dq` (with `S_{d−1} =
2, 2π, 4π` for d=1,2,3). Costly: `O(n_q·n_a)` self-energy evals; the d>1 bubble is
"exact by composition" so fine resolution isn't required for correctness.

### 4.6 `full_integrator._heat_kernel_x_general(Bcal_g, xs_arr, spatial_dim)` — `full_integrator.py:143`
The **production, batched** analytic IFT — the vectorized cousin of
`gaussian_heat_kernel`, operating on a whole batch of per-chamber Schwinger
samples at once.
**Inputs.** `Bcal_g` — either `(P',)` (n_ext=1, k=2: scalar `𝓑` per sample) or
`(P', n, n)` (n_ext≥2, k≥3: matrix `𝓑` per sample); `xs_arr` — the evaluation
points (scalar offsets `(n_x,)` or vectors `(n_x, d)` for n_ext=1; `(n_x, n)` /
`(n_x, n, d)` for n_ext≥2); `spatial_dim`.
**Returns.** `hk` of shape `(P', n_x)`.
**The math it implements.**
- n_ext=1: `hk = (4πB)^{−d/2} e^{−|x|²/4B}` — exactly `gaussian_heat_kernel`'s
  body without the `θ(t)`/drift/time pieces (those are folded into `B` =
  `D·Q_eff` per Schwinger sample upstream). `:164-170`
- n_ext≥2: the **multivariate Gaussian IFT** `hk = (4π)^{−nd/2} det(𝓑)^{−d/2}
  exp(−¼ Σ_c X⃗_cᵀ𝓑⁻¹X⃗_c)`, one factor per spatial component `c`, sharing `𝓑`
  by isotropy. Uses `np.linalg.det`, `np.linalg.inv`, and `np.einsum` to contract
  the quadratic form. `:171-184`. Callers must pre-filter degenerate samples
  (`det 𝓑 > 0`).

This is what `pipeline_bridge.diagram_correlator_x` (the analytic-IFT branch)
calls per diagram per chamber to get `δC(x, τ)` directly with **no q-grid**.

---

## 5. Data structures

### 5.1 The spatial block of the prop dict (output of `build_spatial_propagator`)
A plain, **picklable** dict (`heat_kernel.py:421-431`):

| key | type | meaning |
|---|---|---|
| `G_tx_sym` | `dict[(i,j)] -> (A_expr, B_expr) or None` | per-entry symbolic mass/diffusion; `None` off-diagonal |
| `k_var` | SR var | the momentum symbol `k` |
| `spatial_dim` | int (1/2/3) | spatial dimension `d` |
| `bc_mode` | str | `'infinite'` or `'periodic'` |
| `bc_params` | dict | boundary params (e.g. `{'length': 'L'}`); excludes `'mode'` |
| `initial_mode` | str | `'stationary'` (default) etc. |
| `ac_mass` | `dict[i] -> A_expr` | per-field mass (SR) |
| `ac_diffusion` | `dict[i] -> B_expr` | per-field diffusion (SR) |
| `ac_drift` | `dict[i] -> V_expr` | per-field drift (SR); `0` = pure heat kernel |

The runtime callables are deliberately **not** stored here (closures don't
pickle); they are rebuilt by `make_g_tx_callables`. Note `G_tx_sym` stays a
2-tuple `(A, B)` for backward compatibility; the drift travels separately in
`ac_drift`.

### 5.2 The `G_tx` runtime dict (output of `make_g_tx_callables`)
`dict[(i,j)] -> callable(t, x, **num_params) -> complex`. Off-diagonal /
`None`-symbol entries return `0j`.

### 5.3 The correlator output `(C, info)`
- `C`: `np.ndarray`, shape `(len(tau_grid), len(spatial_grid))`, dtype
  `complex128`.
- `info` (tree path): `{'field_index', 'A_mass', 'B_diffusion', 'D_noise',
  'bc_mode', 'L'}`.
- `info` (pipeline path): `{'field_index', 'modes', 'bc_mode', 'L',
  'spatial_dim', 'pipeline_certified', 'certify_max_rel'}`.
- `info` (coupled path): `{'coupled': True, 'M', 'D0', 'Dhat', 'N', 'legs',
  'bc_mode', 'L', 'spatial_dim', ...}`.

### 5.4 The per-edge routing triple (oracle path)
An edge is `(a, b, kind)`: `a` = length-`L` loop-momentum coefficient tuple, `b`
= length-`n_ext` external-momentum coefficient tuple, `kind ∈
{'R','retarded','C','correlation'}`. The momentum is `k_e = Σ_i a_{ei}ℓ_i + Σ_j
b_{ej}q_j`.

### 5.5 The Symanzik matrices
`Lam` `(L,L)`, `N` `(L, n_ext)`, `Q` `(n_ext, n_ext)`; `U = det(Lam)` (scalar);
`Q_eff = Q − Nᵀ Lam⁻¹ N` `(n_ext, n_ext)`. The per-sample batched `𝓑` is
`D·Q_eff` (scalar `(P,)` for n_ext=1, matrix `(P, n, n)` for n_ext≥2).

### 5.6 The mode list (`modes`, pipeline path)
A list of `(mu, D, kap)` triples — one per heat-kernel mode — read from the
propagator. For v1's tree + constant-mass-shift scope the dressed propagator
stays single-mode.

---

## 6. Data flow (concrete)

### 6.1 Building the propagator block
**In:** `K_ft` (Sage matrix), `omega` (SR), `ns` (with `ns.Laplacian`), `model`
(with `model['spatial']['dim']`, optional `model['boundary'] = {'mode':
'periodic', 'length': 'L'}`), `resp_names`, `phys_names`.
**Out:** the spatial-block dict (§5.1). Example diagonal entry → `extract_mass_diffusion`
peels `K_ft[0,0] = mu + 3*lambda*phistar1^2 − D*Laplacian + I*omega` into
`A = mu + 3*lambda*phistar1^2`, `B = D`, `V = 0`.

### 6.2 Reconstructing the runtime callables
**In:** the prop dict (fresh build or cache load).
**Out:** `G_tx[(i,j)]` closures. Calling `G_tx[(0,0)](t=0.5, x=1.0, mu=1.0,
lambda=0.0, phistar1=0.0, D=0.1)` returns the heat-kernel value `(4π·0.1·0.5)^{−1/2}
e^{−1/(4·0.1·0.5) − 0.5}`.

### 6.3 Tree correlator
**In:** `(ft, model, prop, num_params, external_fields=[('phi',1),('phi',1)],
tau_grid, spatial_grid)`.
**Out:** `(C[τ,x], info)`. Internally: resolve field index → read `(A,B)` →
sign-check `B` → read `D_noise` from the noise sector → fill `C[it,ix] =
free_two_point(A, B, Dn, x, τ, ...)`.

### 6.4 Production correlator via the pipeline bridge
**In:** the same plus `q_samples`, `tau_samples`, `certify_tol`, `q_cut`, `n_q`.
**Out:** `(C[τ,x], info)`. Internally: read `modes` from the propagator →
(optionally) certify against the diagram `C(q,τ)` → **d=1:** `C[it,ix] = Σ_modes
free_two_point(...)`; **d≥2:** build the radial `Cq = Σ_modes (κ/m) e^{−m|τ|}`
on a `q_cut`-truncated grid and apply `radial_inverse_ft` (or
`periodic_inverse_ft` on a box).

### 6.5 Loop correction (full integrator)
**In:** the live diagram list, `mu0, D0, kap0`, the `(τ, x)` grids, env knobs.
**Out:** `δC(x,τ)` accumulated per loop order (`dCx_by_ell`). If `_use_analytic`
(`pipeline_bridge.py:1677`), each diagram's per-chamber Schwinger samples produce
`𝓑 = D·Q_eff`, fed to `_heat_kernel_x_general` for the analytic IFT — **no
q-grid**. Otherwise the numerical FT path builds `δC(q,τ)` on the q-grid and
transforms with `radial_inverse_ft`.

---

## 7. Gotchas & caveats

### 7.1 Environment-variable knobs
All three knobs are read in `pipeline_bridge.py` (not in the four primary files —
they are the *gate* that selects the analytic vs numerical path):

- **`SPATIAL_FORCE_NUMERICAL_FT`** — `pipeline_bridge.py:610` and `:1676`. When
  set (`== '1'` at `:1676`; truthy at `:610`), forces the **numerical FT path**
  even when the analytic heat-kernel IFT would apply. This keeps the
  cosine/radial-FT cross-check reachable. It is **exact only in the `q_cut → ∞ /
  n_q → ∞` limit** — the analytic path has no such truncation. Used for
  validation, not production.
- **`SPATIAL_Q_CUT`** — `pipeline_bridge.py:1604`. Overrides the q-grid upper
  cutoff `q_cut` for the numerical-FT cross-check. The comment warns: "derivative-
  vertex form factors with a q⁴ tail need a large q_cut for the truncated trapz to
  converge; the analytic path has no such limit."
- **`SPATIAL_N_Q`** — `pipeline_bridge.py:1605`. Overrides the number of q-grid
  points for the numerical-FT cross-check.

The **gate** itself: `_use_analytic = (_all_plain or d == 1) and not _force_num`
(`pipeline_bridge.py:1677`). I.e. the analytic heat-kernel IFT covers **(a) plain
vertices at any d (Phase 1)** and **(b) any vertices at d=1 (Phase 2 —
derivative-vertex form factors via poly-fit + closed-form q-moments)**. The one
gap is **d≥2 derivative vertices → still numerical** (Phase 3, unimplemented).

Other related knobs nearby (not part of this subsystem proper but affect the same
code path): `SPATIAL_GRID_NT`/`SPATIAL_GRID_NS` (chamber grid sizes),
`SPATIAL_INTEGRATOR` (`grid`/`mc`/`bessel`), `SPATIAL_MC_N`,
`SPATIAL_MEM_BUDGET_GB` (the OOM memory guard at `:1650`).

### 7.2 Tier-1 only: the diagonal restriction
The closed-form heat-kernel path is **Tier 1**: it requires a **diagonal**
inverse propagator (each field an independent Allen-Cahn mode). Any off-diagonal
coupling makes `build_spatial_propagator` raise (`heat_kernel.py:369-377`) and
leaves `G_tx_sym = None`. Downstream, `compute_spatial_correlator_tree` raises a
clear `NotImplementedError` (`spatial_correlator.py:239-248`), and the pipeline
bridge re-routes to the coupled spectral-Lyapunov driver
(`compute_coupled_tree_correlator`) when possible.

### 7.3 Higher-derivative operators are rejected
`extract_mass_diffusion` rejects any k-power > 2 (`heat_kernel.py:213-216`) — a
`Laplacian²` (k⁴, e.g. conserved Model-B in the *kinetic* operator) is a v2
feature. Note: conserved (q²-dependent) **noise** is separately skipped by
`extract_noise_matrix` (left 0).

### 7.4 The `iω` normalization is strict
The time-derivative coefficient must be **exactly `i`** (`heat_kernel.py:201`,
`:263`). A rescaled time derivative (non-unit `iω`) is rejected as "not a standard
MSR kernel." This is a deliberate guard, but it means a model that hides a factor
on `∂_t` will fail here rather than silently rescale.

### 7.5 Diffusion sign / zero guards
`compute_spatial_correlator_tree` raises on `B < 0` (anti-diffusive, ill-posed,
the heat kernel diverges — check the sign of `−D·Laplacian` with `D > 0`) and on
`B == 0` (no Laplacian → a time-only field; a spatial correlator is undefined —
"request this field's correlator without a spatial_grid"). `:261-275`. The
underlying reason: the erf form divides by `√(4πB)` and takes `√` of `α` —
negative/zero `B` breaks both.

### 7.6 PBC and drift are d=1 only
- `image_sum` raises for `d ≠ 1` (`heat_kernel.py:143-145`).
- `free_two_point`'s periodic branch is d=1 (the periodic FT for d≥2 is the
  separate `periodic_inverse_ft` lattice sum).
- `gaussian_heat_kernel` raises if `V ≠ 0` and `d ≠ 1` (`heat_kernel.py:85-87`) —
  drift is a vector in higher d.
- `build_spatial_propagator` raises if `d ≠ 1` and the boundary is periodic
  (`heat_kernel.py:345-349`).

### 7.7 Drift is usually zero — and that's correct
For Burgers/KPZ the only `∂_x` reaching the bilinear sector is `∝ φ*`, which
vanishes at the homogeneous saddle `φ* = 0`, so `V → 0` and the propagator is the
**pure heat kernel**. Don't be surprised that a "KPZ" propagator has no drift at
the trivial saddle — that is physically right.

### 7.8 The erf cancellation needs mpmath precision
`erf_time_integral` runs at 30 decimal digits **on purpose**: the
`e^{+2√(αβ)}·erf + e^{−2√(αβ)}·erf` form has a near-cancellation that double
precision cannot resolve for large `√(αβ)`. If you ever port this to plain
floats, expect it to silently lose digits.

### 7.9 The `w → 0⁺` limit in `erf_time_integral`
The helper `_F(w)` special-cases `w == 0` (`heat_kernel.py:116-120`): `erf(±∞) =
±1` when `β > 0`, `erf(0) = 0` when `β = 0`. This is the `s → 0` endpoint when
`L_lo == 0`. Getting this wrong would produce a `0/0` or a spurious value at the
`x = 0`, equal-time corner.

### 7.10 Numerical-FT truncation (Regime 1)
Both `radial_inverse_ft` and `periodic_inverse_ft` **truncate at `k_max =
q_grid[−1]`** — they evaluate the correlator *at* a finite momentum cutoff, not
in the continuum. The continuum value is the `k_max → ∞`, fine-grid limit. This
is the "Regime 1" physical-cutoff interpretation. The **analytic** heat-kernel
IFT has no such truncation (this is the whole point of "retiring the q-grid").

### 7.11 Memory guard for ℓ≥2 (sibling, but reached on this path)
`pipeline_bridge.py:1650-1665` refuses up-front to allocate a chamber's
`n_t^{n_V}·n_s^{n_C}·n_x` complex array if it would exceed
`SPATIAL_MEM_BUDGET_GB` (default 6 GB) — a deliberate guard against an OOM that
"crashes the kernel AND the OS" (a documented macOS-fork-adjacent hazard). Not in
the four primary files, but it governs whether the analytic IFT even runs.

### 7.12 ORACLE-ONLY modules must not be presented as production
`temporal_integrate.py` and `loop_parametric.py` are cross-check oracles only;
`compute_cumulants` does not call them. Their banner says so explicitly. The
`sigma_K_kernel` `dblquad` is even noted as "the C-stack's bottleneck" that the
production-adjacent `sigma_parametric` Gauss–Laguerre rule was written to replace.

### 7.13 Argument-name aliasing across functions
`free_two_point(mu, D, kap, …)` is called as `free_two_point(A, B, Dn, …)` in
`compute_spatial_correlator_tree` (`:291`). This is *semantically correct* (`A` =
mass → `mu`; `B` = diffusion → `D`) but the name change can read as a bug at a
glance. Likewise the bridge calls it as `free_two_point(mu, D, kap, …)` with the
mode's own `(mu, D, kap)`. Always remember: first arg = mass, second = diffusion.

---

## 8. Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the response-functional
  formalism that turns a stochastic PDE into a field theory with a physical field
  `φ` and a response ("tilde") field `φ̃`.
- **Inverse propagator (`K_ft`)** — the quadratic-form matrix of the free action
  in `(ω, k)`; its inverse is the propagator `G`. Here it has the Allen-Cahn form
  `A + B·k² + i·ω`.
- **Propagator `G`** — the two-field Green's function; the response/retarded
  block is `G_R`.
- **Retarded propagator `G_R`** — the causal response function, `∝ θ(t)`; zero for
  `t < 0`.
- **Allen-Cahn / reaction-diffusion** — a field with a relaxational mass `μ` and a
  diffusion `D∇²`; the prototypical Tier-1 model.
- **Mass `A` (a.k.a. `μ`)** — the `k⁰` term of the inverse propagator; the
  relaxation rate.
- **Diffusion `B` (a.k.a. `D`)** — the `k²` coefficient; physically positive.
- **Drift `V`** — the `k¹` (advection) coefficient; `i·v` for velocity `v`; zero
  for even/Laplacian-only kernels.
- **Heat kernel** — the diffusion-equation Green's function `(4πBt)^{−d/2}
  e^{−x²/4Bt}`; the inverse FT of a momentum-space Gaussian.
- **IFT (inverse Fourier transform)** — here, the `k → x` (or `q → x`) transform.
  "Analytic IFT" means done in closed form (no q-grid).
- **q-grid** — a discretized momentum axis used by a *numerical* FT; "retired" by
  the analytic heat-kernel IFT for plain/d=1 cases.
- **Schwinger parameter** — an auxiliary integration variable `w_e` per edge that
  exponentiates a propagator denominator; here it is literally the edge's time
  duration in the heat-kernel `(k,t)` representation.
- **Symanzik polynomials (`U`, `F`/`Q_eff`)** — the two graph polynomials that
  result from doing a Feynman/loop momentum integral in Schwinger parameters; `U
  = det(Lam)`, `Q_eff = Q − Nᵀ Lam⁻¹ N`.
- **Routing coefficients `(a_e, b_e)`** — how each edge's momentum decomposes into
  loop momenta `ℓ` and external momenta `q`: `k_e = Σ a_{ei}ℓ_i + Σ b_{ej}q_j`.
- **Self-energy `Σ`** — the 1PI two-point correction; dresses the propagator via
  the Dyson equation.
- **Keldysh / correlation `C` edge vs retarded `R` edge** — a `C` edge carries a
  noise-driven correlation (Schwinger param integrated over `[|t|,∞)`); an `R`
  edge is a fixed-duration retarded propagator.
- **`𝒮(Γ)` (symmetry/combinatorial factor)** — the diagram automorphism factor;
  applied by the caller, not by `sigma_parametric`.
- **erf-split closed form (Rescue A)** — the antiderivative of `s^{−1/2}e^{−β/s−αs}`
  written as a sum of two `erf`s with `e^{±2√(αβ)}` weights.
- **Image sum / method of images** — the periodic-BC Green's function as a sum of
  displaced infinite-domain kernels.
- **Tier 1 vs Tier 2** — Tier 1 = the diagonal closed-form heat-kernel path; Tier
  2 = the numerical / coupled / higher-derivative fallback (v2).
- **Regime 1 (finite cutoff)** — evaluating the correlator *at* a finite momentum
  cutoff `k_max`; the continuum is `k_max → ∞`.
- **Bigrade / `(2,0)` sector** — the action's term with 2 response fields and 0
  physical fields, i.e. the white-noise covariance `−D^noise φ̃²`.
- **SR (Symbolic Ring)** — SageMath's computer-algebra engine for symbolic
  expressions.
- **mpmath** — arbitrary-precision floating-point + special functions (used for
  the erf cancellation).
- **Gauss–Laguerre quadrature** — a numerical rule that integrates `∫₀^∞ e^{−x}
  g(x) dx` exactly for polynomial `g`; used as a tensor rule over Schwinger
  params in the oracle path.
- **Hankel transform / `J₀`** — the radial Fourier transform in d=2 (uses the
  Bessel `J₀`).
- **`K₀`** — the modified Bessel function appearing in the d=2 static correlator.
- **ORACLE-ONLY** — a module kept solely as an independent numerical cross-check;
  not on the production path.

---

## 9. Proposed manual subsections

1. **From the inverse propagator to real space** — the Allen-Cahn kernel `A +
   B·k² + i·ω`, closing the ω-contour, and why `G_R(k,t) = θ(t)e^{−(A+Bk²)t}`.
2. **The heat kernel** — the `k → x` Gaussian IFT, complex `A`/`B`, the
   d-dependence in the `(4πBt)^{−d/2}` prefactor; `gaussian_heat_kernel`.
3. **Drift and advection** — the `k¹` term, the centre-shift `x = vt`, why it
   vanishes at the homogeneous saddle (KPZ/Burgers), and the d=1 restriction.
4. **Symbolic extraction** — reading `(A, B, V)` off `K_ft` with Sage:
   `extract_mass_diffusion`, the operator substitutions, and the MSR-normalization
   / higher-derivative guards. (Include the coupled-field `reaction_diffusion_matrices`.)
5. **Periodic boundaries: the image sum** — method of images, super-exponential
   truncation, d=1 scope; `image_sum`.
6. **The tree-level correlator** — noise driving two retarded legs, the
   semigroup collapse, the erf-split closed form (`erf_time_integral`,
   `free_two_point`), and the static `τ=0` oracle.
7. **Noise extraction** — the `(2,0)` sector, the `D_noise` / `N` conventions
   (`extract_noise_coefficients`, `extract_noise_matrix`).
8. **The d≥2 output transform** — radial / Hankel / lattice inverse FTs
   (`radial_inverse_ft`, `periodic_inverse_ft`) and the finite-cutoff (Regime 1)
   caveat.
9. **From propagator to runtime: build vs reconstruct** — the picklable spatial
   block, `make_g_tx_callables`, and why closures aren't cached.
10. **The analytic IFT for loops** — the per-chamber Gaussian `𝓑 = D·Q_eff`, the
    batched `_heat_kernel_x_general`, and the `_use_analytic` gate (plain any-d /
    derivative d=1; the d≥2-derivative gap).
11. **The numerical-FT cross-check and the env knobs** —
    `SPATIAL_FORCE_NUMERICAL_FT`, `SPATIAL_Q_CUT`, `SPATIAL_N_Q`, and how the
    analytic path is validated against the truncated q-grid FT.
12. **The Symanzik momentum core (oracle path)** — `spatial_reduce`,
    `loop_parametric`, `temporal_integrate`; what they validate and why they are
    not on the production path.
13. **Guards, errors, and known limits** — `SpatialPropagatorError`, the diagonal
    Tier-1 restriction, diffusion sign/zero, the strict `iω` normalization, the
    memory guard, and the v2 backlog (off-diagonal coupling, higher-derivative
    operators, d≥2 derivative vertices).

---

## Appendix: cross-references and source pinpoints

- Heat kernel core: `heat_kernel.gaussian_heat_kernel` (`heat_kernel.py:58`).
- Batched production IFT: `full_integrator._heat_kernel_x_general`
  (`full_integrator.py:143`).
- The analytic-vs-numerical gate: `_use_analytic` (`pipeline_bridge.py:1677`).
- Env knobs: `pipeline_bridge.py:610, 1604, 1605, 1676`.
- Validation spike (referenced by docstrings, not read here):
  `docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py`.
- Authoritative pipeline doc: `docs/spatial_pipeline.md`; analytic-IFT plan:
  `docs/spatial_analytic_ift_plan.md`.

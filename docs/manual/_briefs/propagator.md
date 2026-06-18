# Propagator Construction (frequency domain, poles, residues)

**Subsystem slug:** `propagator`

**Primary source files**

| File | Role |
|------|------|
| `pipeline/_propagator.py` | The whole engine: builds `K_ker → K_ft → G_ft`, finds retarded poles, computes residue matrices `C_mats`, and the instantaneous-limit matrix `D_delta`. |
| `msrjd/core/propagator.py` | **A stub only.** A 10-line docstring saying "logic currently lives in the demo notebook." Nothing here is imported by the working pipeline; the real code is `pipeline/_propagator.py`. (Recorded in *open_questions* below — it is a stale placeholder.) |
| `msrjd/integration/time_domain/propagator_td.py` | Turns the frequency-domain propagator data (poles + residues + `D_delta`) into a **time-domain** propagator matrix `G(t)` = smooth pole-residue sum + δ(t) part, and offers per-edge lookups `G_t_entry` / `G_t_delta_coeff` used by the diagram integrators. |

Supporting code that this brief touches because the primary files call into it:

- `msrjd/core/field_theory.py` — `fourier_transform()` (the FT convention), and the `FieldTheory` accessors `ring()`, `free_action()`, `_n_tilde`, `_ns`.
- `msrjd/core/cache.py` — `PipelineCache`, the on-disk `.sobj` cache.
- `msrjd/integration/spatial/heat_kernel.py` — `build_spatial_propagator()` / `make_g_tx_callables()` (only invoked when the model declares a spatial field; mentioned for completeness).

---

## Overview

### What this subsystem does, in plain language

In an MSR-JD (Martin–Siggia–Rose–Janssen–De Dominicis) field theory the **free (Gaussian) part of the action** is a bilinear form coupling each *response* field `φ̃ᵢ` to each *physical* field `φⱼ`. Collect the coefficients of that bilinear form into a matrix **K** (the "kinetic" or "kernel" matrix). The **propagator** is its inverse, **G = K⁻¹**. Every Feynman diagram in the perturbative expansion is built by wiring vertices together with copies of `G`, so *nothing downstream can run until `G` exists.*

Because the free action contains time derivatives `∂_t` and delta-kernels `δ(t)`, `δ′(t)`, the natural place to invert `K` is **frequency space**: Fourier-transform `K` in time to get `K_ft(ω)`, invert the (small, e.g. 2×2 or 4×4) matrix once, and read off

```
G_ft(ω) = K_ft(ω)⁻¹
```

`G_ft(ω)` is a matrix of **rational functions of ω**. Its physical content is entirely in:

- the **poles** ω_k (zeros of det K_ft), which set the relaxation/oscillation rates, and
- the **residue matrices** `C_mats[k]` at each pole, which set the amplitudes,
- plus an **instantaneous piece** `D_delta = lim_{ω→∞} G_ft(ω)` that becomes a `δ(t)` in time (the immediate response of `δn` to its own conjugate source `ñ` in the MSR-JD action).

The time-domain propagator is then the closed-form inverse Fourier transform

```
G_R(t) = D_delta · δ(t) + Θ(t) · Σ_k C_mats[k] · exp(i·ω_k·t)
```

with `Θ(t)` the Heaviside step enforcing *retardation* (response only after the cause). This subsystem produces every object on the right-hand side.

### Where it sits in the end-to-end pipeline

```
   FieldTheory (expanded action)              model dict (params, kernel images)
            │                                          │
            └──────────────┬───────────────────────────┘
                           ▼
        pipeline/_propagator.py : build_propagator()
            K_ker → K_ft → (symbolic) G_ft / adj_ft / D_omega / D_delta
                           │  (cached to saved_theories/<tag>/propagator.sobj)
                           ▼
        pipeline/_propagator.py : compute_poles_and_residues(prop, num_params)
            fills prop['pole_vals'], prop['C_mats'], (and prop['D_delta'])
                           │
                           ▼
   propagator_td.py : build_G_t_matrix(prop, t)  →  {'smooth', 'delta', 't_var'}
            per-edge G_t_entry() / G_t_delta_coeff()
                           │
                           ▼
        diagram tree integrators (final_integral.py, grouped_integral.py)
            wire G(t) onto every edge → correlators / cumulants
```

- **Feeds it:** the *expanded* `FieldTheory` `ft` (from the action-expansion stage in `compute.py` step 1) and the `model` dict (parameters, and the `kernel_ft_image` hook that maps abstract kernel symbols to their frequency images). `compute_poles_and_residues` additionally needs `num_params` — the numeric parameter substitution produced by the mean-field solve (`compute.py` step 3).
- **Consumes its output:** `propagator_td.build_G_t_matrix` (and through it the time-domain tree integrators `final_integral.py`, `grouped_integral.py`); the spatial bridge `spatial/pipeline_bridge.py` also calls `compute_poles_and_residues`.

Call sites in `compute.py`:

- `pipeline/compute.py:357` — `prop = build_propagator(ft, model, use_cache=use_cache, verbose=verbose)` (step 2/7).
- `pipeline/compute.py:632` — `compute_poles_and_residues(prop, num_params, verbose=verbose)` (step 4/7).

A separate precompute entry point at `pipeline/_precompute.py:168` calls `build_propagator(...)` at Taylor order 2 just to warm the cache (the propagator depends only on the bilinear (1,1) sector, which is fully captured at order ≥ 2 — see the docstring of `build_propagator`).

---

## The math

This section builds the theory from the ground up so the equations the code implements are unambiguous.

### 1. The MSR-JD free action and the kernel matrix K

A stochastic field theory written in MSR-JD form has an action `S[φ, φ̃]`. The fields split into two halves of equal count `nf`:

- **physical fields** `φⱼ`, `j = 1..nf` (e.g. a density `n`, a membrane potential `v`),
- **response (auxiliary / "tilde") fields** `φ̃ᵢ`, `i = 1..nf`.

The **free action** is the part of `S` that is bilinear with exactly one response field and one physical field — bigrade **(1,1)** in the code's language (one tilde leg, one physical leg). In `FieldTheory` this is exactly what `ft.free_action()` returns: `self._by_tp.get((1,1), 0)` (`msrjd/core/field_theory.py:1010`). Schematically

```
S_free = Σ_{i,j} ∫dt  φ̃ᵢ(t) · K̂_{ij}(∂_t) · φⱼ(t)
```

where each `K̂_{ij}` is a differential/kernel operator in time. For a simple Ornstein–Uhlenbeck-like field `K̂ = ∂_t + μ` so `S_free ⊃ ∫ ñ (∂_t n + μ n)`.

Reading off the coefficient of `φ̃ᵢ·φⱼ` in `S_free` gives the **kernel matrix K** with layout **[response row, physical col]**:

```
K[i, j] = coefficient of (φ̃ᵢ φⱼ) in S_free.
```

This is precisely the `K_data[row][col]` assembly loop in `build_propagator` (`pipeline/_propagator.py:595-606`).

### 2. Kernel form: δ and δ′

In the time domain a coefficient like `∂_t n` is represented with an abstract "δ-prime" symbol so that the Fourier transform is clean. The code uses three abstract namespace symbols:

- `Dt` (`ns.Dt`) — a marker for the time-derivative operator `∂_t`,
- `delta_D` (`ns.delta_D`, displayed as δ) — the Dirac delta `δ(t)`,
- `delta_Dp` (`ns.delta_Dp`, displayed as δ′) — its derivative `δ′(t)`.

The helper `_to_kernel` (`pipeline/_propagator.py:507`) rewrites every `K[i,j]` into **kernel form** `c₀·δ + c₁·δ′`:

```
c  →  c0·δ_D  +  (c − c0)|_{Dt → δ_Dp}     where c0 = c|_{Dt → 0}
```

So a plain constant `μ` becomes `μ·δ(t)` (an instantaneous coupling) and a derivative `Dt` becomes `δ′(t)`. Wrapping the *constant* part in `δ(t)` is what makes its Fourier transform return the constant back rather than a `2π·δ(ω)` distribution (see the docstring at `:514`). Abstract kernel symbols (e.g. `ns.g`, a synaptic kernel) are left untouched here; they are transformed later via the model's `kernel_ft_image` hook.

### 3. Fourier transform to K_ft(ω)

The pipeline-wide Fourier convention (defined in `msrjd/core/field_theory.py:33`, `fourier_transform`) is the **angular-frequency, no-2π-in-exponent** convention:

```
F(ω) = ∫_{-∞}^{∞} f(t) · e^{−iωt} dt
```

Under this convention:

```
δ(t)  →  1
δ′(t) →  iω
```

So a kernel-form entry `c₀·δ + c₁·δ′` Fourier-transforms to `c₀ + c₁·(iω)`. For the OU example `K̂ = ∂_t + μ` becomes `K_ft = iω + μ`. The replacement is done by substituting the abstract δ symbols for their concrete Sage forms (`dirac_delta(t)` and `diff(dirac_delta(t), t)`) and calling `fourier_transform` per nonzero entry (`pipeline/_propagator.py:625-639`). After the FT the model's `kernel_ft_image(ns, omega)` hook replaces any remaining abstract kernel symbol `g` with its frequency image `ĝ(ω)` (`:642-645`).

**Why poles land in the upper half plane.** Because the FT uses `e^{−iωt}`, the *inverse* FT closes the ω-contour in the **upper** half plane for `t > 0`. Hence the *retarded* (causal) propagator is parameterized by poles with **Im(ω) > 0**. This is the single most important sign convention in the subsystem and is restated in `propagator_td.py:12-20`.

### 4. The propagator as a matrix inverse

```
G_ft(ω) = K_ft(ω)⁻¹
```

Two companion objects:

- **adjugate** `adj_ft = adj(K_ft)`. By Cramer's rule `K⁻¹ = adj(K)/det(K)`, so `adj_ft = G_ft · det(K_ft)`. The code computes it as `adj = G * D_om` to *reuse* the inverse's work rather than re-expanding cofactors (`pipeline/_propagator.py:710-715`; the comment notes `adjugate()` measured 734 s vs ~11 s for the inverse on a 4×4 spike-reset matrix).
- **characteristic determinant** `D_omega = det(K_ft)`. Its numerator `N(ω)` is the **characteristic polynomial**; its roots are the system poles.

`G_ft[i,j]` is a rational function `P_ij(ω)/Q(ω)`. For a diagonally-dominant `K_ft` every entry shares the same denominator `Q(ω)` = polynomial numerator of `det(K_ft)`.

### 5. Poles and residues

Let ω_k be a simple zero of `Q(ω)` (equivalently of `num(det K_ft)`) with **Im(ω_k) > 0**. The residue of `G_ft` there is

```
Res_{ω=ω_k} G_ft(ω) = adj(K_ft(ω_k)) / det′(K_ft)(ω_k)        (entrywise: P_ij(ω_k)/Q′(ω_k))
```

The code stores the **residue matrix scaled by i**:

```
C_mats[k] = i · Res_{ω=ω_k} G_ft(ω)            (the "C_k convention")
```

The factor of `i` makes the inverse-FT come out clean. Per entry (polynomial path), `C_k[i,j] = i · P_ij(ω_k) / Q′(ω_k)` (`pipeline/_propagator.py:97-98`, implemented at `:268`). In the numerical-cofactor path, `C_np = 1j * adj_at_pk / d_prime` (`:1496`).

### 6. The instantaneous limit D_delta

If `G_ft(ω)` does **not** vanish as ω→∞ (a *non-proper* rational function), the inverse FT produces a `δ(t)` term. Decompose

```
G_ft(ω) = Q_poly(iω) + G_proper(ω),    G_proper strictly proper (→0 as ω→∞)
⟹  G(t) = Q_poly(∂_t) δ(t) + Θ(t)·[residue sum]
```

For the common case where the polynomial part is a constant matrix,

```
D_delta[i, j] = lim_{ω→∞} G_ft[i, j](ω).
```

Concretely, per entry with `G_ft[i,j] = P_ij/Q`:

- `deg(P_ij) <  deg(Q)`  → `D_delta[i,j] = 0`,
- `deg(P_ij) == deg(Q)`  → `D_delta[i,j] = lc(P_ij)/lc(Q)` (ratio of leading coefficients).

This is the textbook rational-limit recipe (`_omega_inf_limit_fast`, `:460`; polynomial-path version `:283-295`). Physically, `D_delta` is the immediate, same-time response — e.g. an `ñ` source at time `t` produces a `δn` at the *same* `t` (`propagator_td.py:147-150`).

### 7. The full time-domain propagator

Putting it together (the inverse FT of a sum of simple poles plus a constant):

```
G_R[i, j](t) = D_delta[i, j] · δ(t) + Θ(t) · Σ_k C_mats[k][i, j] · exp(i · ω_k · t)
```

This is exactly `build_G_t_matrix`'s output: a `smooth` SR matrix `Σ_k C_k·exp(i·p_k·t)` plus a `delta` matrix (`propagator_td.py:135`, assembled at `:285-287`). With `Im(ω_k) > 0`, each `exp(i·ω_k·t)` decays for `t>0` (retarded) and grows for `t<0` — the `Θ(t)` makes it well-defined.

### 8. Index transpose convention

K and G both have layout `[resp_row, phys_col]`. But the *physical* retarded propagator is "response of physical field j to a response-field source i," `G^R_{j←i}`, which means reading `G[j, i]` — the **transpose**. `G_t_entry(phys_idx=j, resp_idx=i)` reads `smooth[phys_idx, resp_idx]` to apply that transpose (`propagator_td.py:44-50, 425`). Every consumer must use the same transpose; this one matches `_get_propagator_entry` in `msrjd/integration/symbolic.py`.

---

## External tools used

This subsystem leans almost entirely on **SageMath** (and through it, several computer-algebra back-ends), with **NumPy** for the numerical fallback paths and **cysignals** for interrupting runaway native routines. None of nauty, networkx, numba, or sympy is called *directly* here — though Sage's `fourier_transform` routes through SymPy internally (see below).

### SageMath (`sage.all`)

**What it is.** SageMath is a large open-source mathematics system that bundles dozens of specialized libraries (Pari, Singular, Maxima, FLINT, GMP, NumPy, SymPy, …) behind one uniform Python API. Think of it as "Mathematica, assembled from open-source parts, scriptable in Python." When you `import` from `sage.all`, you get symbolic expressions, exact number fields, polynomial rings, matrices over arbitrary rings, and the glue that routes each operation to the right back-end.

**Import lines.**

```python
# pipeline/_propagator.py:23-26
from sage.all import (
    SR, matrix, dirac_delta, diff, oo, limit as _sage_limit,
    QQ, CDF, PolynomialRing, CyclotomicField,
)
```
```python
# propagator_td.py:53
from sage.all import SR, I, exp, heaviside, matrix, CDF
```
```python
# core/cache.py
from sage.all import save as sage_save, load as sage_load
```

The pieces of Sage this code uses, each explained:

- **`SR` — the Symbolic Ring.** Sage's general symbolic-expression type (a wrapper around the GiNaC/Pynac C++ engine). `SR(x)` coerces anything to a symbolic expression; `SR.var('omega')` makes a free symbol. Almost every matrix here lives over `SR`. Operations like `.numerator()`, `.denominator()`, `.coefficients(omega, sparse=False)`, `.subs(...)`, `.degree(var)`, `.is_zero()`, `.variables()`, `.factor()`, `.simplify_full()` are SR methods. Caveat: SR routes many operations through **Maxima** (a Lisp CAS) and **Singular** (a polynomial CAS), both of which can be pathologically slow or even hang — hence the timeout machinery (see Gotchas).

- **`matrix(ring, data)`** — builds a Sage matrix over a given ring. Used as `matrix(SR, ...)` (symbolic matrices), `matrix(CDF, ...)` (complex-double matrices for residue output), and `matrix(F_exact, ...)` (over the exact polynomial fraction field). Matrix methods used: `.inverse()`, `.det()`, `.adjugate()`, `.apply_map(fn)` (apply a function entrywise), `.dimensions()`, `.nrows()`, `.ncols()`.

- **`dirac_delta(t)` / `diff(expr, var)`** — Sage's symbolic Dirac delta and differentiation. `diff(dirac_delta(t), t)` is the symbolic δ′(t) used as the concrete image of the abstract `delta_Dp` before the Fourier transform (`pipeline/_propagator.py:627-630`).

- **`oo`** — Sage's symbolic infinity (`∞`), used as an FT integration bound and the `ω→∞` limit point.

- **`limit`** — Sage's symbolic limit (routes through Maxima). Imported as `_sage_limit` in `_propagator.py` (note: it is imported but the file prefers the hand-rolled `_omega_inf_limit_fast` to *avoid* Maxima). In `propagator_td.py:309` it is imported as `_limit` and used as a last-resort `D_delta` computation when no precomputed `D_delta` and only `G_ft` are available.

- **`QQ`** — the field of **rational numbers** ℚ. Used to rationalize float parameters (`QQ(c.real_part())`) so the exact polynomial path can run over an exact field (`pipeline/_propagator.py:124-126`).

- **`CDF`** — **Complex Double Field**: complex numbers backed by C `double` (machine-precision floating point, ≈ 1e-16). Used for numerical root-finding (`Q_cdf.roots(CDF)`) and for storing residue matrices as `matrix(CDF, ...)`. `complex(CDF_value)` converts back to a Python `complex`.

- **`PolynomialRing(ring, 'name')`** — constructs the ring of univariate polynomials in one variable over a given coefficient ring. Three flavors appear:
  - `PolynomialRing(CF, 'om_exact')` — exact polynomials over the cyclotomic field (the heart of the exact residue path).
  - `PolynomialRing(CDF, 'om_cdf')` / `'omega'` — floating-point polynomials for numerical root-finding.
  - `.fraction_field()` turns `R[ω]` into `Frac(R[ω])` = rational functions `P/Q`; the inverse of a matrix over this field returns each entry in **canonical irreducible** `P_ij/Q` form (because GCD over a field is reliable).
  Polynomial methods used: `.roots(CDF)` (numerical root-finding), `.derivative()`, `.degree()`, `.leading_coefficient()`, `.coefficients(sparse=False)` (dense coefficient list, ascending degree), `.lcm(other)`, `.is_squarefree()`, `.gcd()` (implicitly, in fraction-field reduction), and calling a polynomial like a function `p(pk)` to evaluate it.

- **`CyclotomicField(4, 'I_')`** — the number field ℚ(ζ₄) = ℚ(i), i.e. the rationals extended by a primitive 4th root of unity, which *is* the imaginary unit i. This is the cleverest tool in the file (`pipeline/_propagator.py:115-117`). **Why use it instead of ℂ?** Because polynomials over an *exact* field have reliable GCDs, so when Sage inverts the matrix in `Frac(CF[ω])` it returns each entry in genuinely-reduced form and the denominator is *exactly* the characteristic polynomial — no spurious cancellable factors and no floating-point GCD noise. A `CDF[ω]` (floating-coefficient) path, by contrast, left "degree-18 polynomials with ~13 spurious roots" (`:106-107`). `CF.gen()` is the generator i; `CF(re) + CF(im)*i_CF` builds an exact Gaussian rational.

- **`save` / `load` (as `sage_save` / `sage_load`)** — Sage's pickling. `sage_save(obj, path)` writes a `.sobj` (Sage object) file; `sage_load(path)` reads it back. `PipelineCache.save/load` wrap these. Note: Sage objects (symbolic matrices, etc.) pickle fine, but **closures do not**, which is why the spatial `G_tx` callables are attached *after* the cache save (`pipeline/_propagator.py:817-836`).

- **`I`, `exp`, `heaviside`** (in `propagator_td.py`) — Sage's symbolic imaginary unit, exponential, and Heaviside step. `exp(I·p_k·t)` builds each pole mode; `heaviside(t_expr)` enforces retardation in `G_t_entry`. Note the docstring warning that Sage's `heaviside(0) = 1/2` by default — used only for symbolic display, never in the numerical integrand (`propagator_td.py:30-43`).

### SymPy (indirect, via Sage's `fourier_transform`)

**What it is.** SymPy is a pure-Python symbolic-math library. Sage can dispatch integration to it as an alternative back-end to Maxima.

**How this code uses it.** Not directly — but `fourier_transform` (`msrjd/core/field_theory.py:67`) tries `integrate(integrand, t, a, b, algorithm='sympy')` *first*, falling back to `'maxima'`. SymPy is preferred because it "handles `positive=True` assumptions on time constants automatically" without interactively asking for the sign of ω. Since `build_propagator` calls `fourier_transform` on every nonzero `K_ker` entry, SymPy is on the hot path even though no `import sympy` appears in the propagator files.

### NumPy (`numpy`, imported as `np`)

**What it is.** The standard Python numerical-array library: dense N-dimensional arrays plus fast linear algebra (`numpy.linalg`).

**How this code uses it.** Imported lazily inside functions (`import numpy as np` at `_propagator.py:113`, `:937`). Used throughout the **numerical fallback** paths:

- `np.linalg.inv(K_at)` — invert the numerical kernel matrix at a frequency probe (`_G_at`, `:1069`).
- `np.linalg.det(_K_at(om))` — numerical determinant for Newton refinement and residues (`:1087`, `:1292`, `:1489`).
- `np.delete(M, idx, axis=...)` — drop a row/column to form a minor for the **cofactor adjugate** (`_cofactor_adj`, `:1477`).
- `np.roots(coeffs_desc)` — polynomial roots (only as a fallback when Sage's root-finder fails, `:1023`).
- `np.polyfit(omegas, dets, deg)` — fit a polynomial to sampled `det(K(ω))` values on a circle (the **Vandermonde-polyfit** fallback pole-finder, `:1095`).
- `np.where(|C|>atol, C, 0)`, `np.max(np.abs(...))`, `np.zeros((nf,nf), dtype=complex)` — thresholding/noise-cleanup and matrix construction.
- `np.exp`, `np.linspace`, `np.pi` — building the sample circle.

NumPy is preferred for residue evaluation because LU-based numerical linear algebra avoids the catastrophic sum-of-rationals cancellation that the symbolic `adj_ft.subs(ω=pk)` path suffers near kernel poles (`:1503-1518`).

### cysignals (`cysignals.alarm`, `cysignals.signals`)

**What it is.** A small Cython library (bundled with Sage) that lets Python code interrupt or recover from signals raised inside native C/Cython routines — including SIGALRM timers and the segfault-wrapping `SignalError`. Plain Python `try/except` cannot catch a hang or crash deep inside Singular; cysignals can.

**How this code uses it.** In `_safe_factor` (`pipeline/_propagator.py:441`):

```python
from cysignals.alarm import alarm, cancel_alarm, AlarmInterrupt
alarm(timeout)
try:
    return e.factor()
finally:
    cancel_alarm()
```

`alarm(timeout)` arms a timer that raises `AlarmInterrupt` after `timeout` seconds, forcibly stopping a runaway `Singular` `factor()` loop. The except clause also catches `cysignals.signals.SignalError` (Sage's wrapper around native segfaults) by class-name check, because in some Sage builds it is *not* a subclass of `Exception` (`:451-457`).

### Python standard library: `signal`, `re`, `time`, `warnings`

- **`signal`** (imported as `_signal`, `:46`) — the POSIX signal API. `_run_with_timeout` uses `signal.SIGALRM` + `signal.alarm(n)` to put a wall-clock budget around the *exact polynomial inverse* and the *symbolic inverse block*. Guarded by `_HAS_SIGALRM` (Windows lacks SIGALRM) and degrades to a plain call if unavailable (`:57-78`). This is the lower-level cousin of the cysignals alarm used for factoring.
- **`re`** — one call: `re.sub(r'[^A-Za-z0-9]+', '_', model['name'])` to slugify the model name into a cache directory tag (`:547`).
- **`time`** — `time.perf_counter()` for the wall-time budgets and verbose timing prints.
- **`warnings`** — `build_G_t_matrix` warns when `pole_vals` is empty (`propagator_td.py:271-283`).

---

## Components

Listed file-by-file, in source order. Signatures are quoted from the code.

### `pipeline/_propagator.py`

#### Module-level constants / setup

- **`_POLYNOMIAL_PATH_BUDGET_SEC = 120`** (`:43`) — wall-time budget (seconds) for the exact polynomial fraction-field inverse before falling back to the numpy-cofactor path. Comment explains quad-φ finishes in <1 s but spike-reset's matrix structure can drive Sage's `CF[ω]` inverse past 10 minutes. `None` disables the timeout.
- **`_HAS_SIGALRM`** (`:45-50`) — boolean: whether `signal.SIGALRM` exists on this platform (False on Windows).
- **`class _PolynomialPathTimeout(Exception)`** (`:53`) — sentinel exception raised when a budget expires.

#### `_run_with_timeout(fn, timeout_sec, *args, **kwargs)` — `:57`

- **Takes:** a zero-or-more-arg callable `fn`, a budget in seconds (or `None`), and pass-through args.
- **Returns:** `fn(*args, **kwargs)`'s result if it finishes in time; raises `_PolynomialPathTimeout` on expiry.
- **Steps:** If no SIGALRM or `timeout_sec is None`, just call `fn` directly (no-op timeout). Otherwise install a SIGALRM handler that raises `_PolynomialPathTimeout`, arm `signal.alarm(int(timeout_sec))`, run `fn`, and in a `finally` cancel the alarm (`alarm(0)`) and restore the previous handler. Used to wrap (a) the exact matrix `.inverse()` (`:147`) and (b) the whole symbolic inverse block (`:732`).

#### `_compute_residues_via_polynomial_fracfield(K_ft, omega, num_params, nf, verbose=False)` — `:81`

**The primary (Tier-1) residue path. Exact arithmetic over ℚ(i)[ω].**

- **Takes:** the symbolic `K_ft` matrix, the ω symbol, the `num_params` substitution dict, the matrix size `nf`, a verbose flag.
- **Returns:** `(pole_vals, C_mats, D_delta)` on success; `(None, None, None)` on any failure (so the caller falls back). *Note the 3-tuple — earlier failure returns in this function are also 3-tuples; verified consistent.*
- **Steps:**
  1. Build the exact rings: `CF = CyclotomicField(4, 'I_')` (= ℚ(i)), `PR_exact = PolynomialRing(CF, 'om_exact')`, `F_exact = PR_exact.fraction_field()` (`:115-118`).
  2. Substitute the numeric parameters into `K_ft`: `K_ft_num = K_ft.apply_map(lambda e: SR(e).subs(num_params))` (`:120`). Now each entry is a univariate rational in ω with numeric coefficients.
  3. `_sr_complex_to_CF(c)` (`:122`) converts an SR complex constant to an exact ℚ(i) element by rationalizing real and imaginary parts: `CF(QQ(c.real_part())) + CF(QQ(c.imag_part()))·i_CF`. **This is the step that can fail** (a non-rationalizable parameter throws, triggering the fallback).
  4. `_sr_to_F_exact(e)` (`:128`) converts a whole SR rational entry into `F_exact` by extracting numerator/denominator coefficient lists (in ω) and building `F_exact(PR_exact(n_coeffs)) / F_exact(PR_exact(d_coeffs))`.
  5. Build the exact matrix `K_ft_F` and invert it under a SIGALRM watchdog: `G_ft_F = _run_with_timeout(K_ft_F.inverse, _POLYNOMIAL_PATH_BUDGET_SEC)` (`:140-148`). Sage returns each `G_ft_F[i,j]` as a canonical irreducible `P_ij/Q`.
  6. **Universal denominator** `Q_poly = LCM over all entries of their denominators` (`:172-181`). The comment explains why LCM, not just the first entry's denominator: an upper-triangular `K_ft` (Markovian-embedding preprocessor) produces entries with *different* canonical denominators, so the first nonzero entry would under-count system poles.
  7. **Squarefree guard** (`:196-209`): if `Q_poly` is not squarefree (a higher-multiplicity / Jordan-block pole, e.g. `Q = (iω+μ)²` from a Markovianized OU at `μ = 1/τ_c`), bail with `(None, None, None)` because the single-pole residue formula can't represent the `(τ·exp(−pt))` derivative term. The fallback tier inherits the same limitation but is more robust to near-degeneracy.
  8. Convert `Q_poly` to `CDF[ω]` (`_cf_poly_to_cdf`, `:153`), take its derivative `Q_prime_cdf`, and find roots `Q_cdf.roots(CDF)`. Keep only **retarded** roots `imag > 1e-9`; sort by `(imag, real)` (`:214-217`).
  9. Cache per-entry numerator/denominator CDF polynomials (`P_cdf`, `Q_per_entry`, `Q_entry_cdf`, `Q_entry_prime_cdf`, `:219-244`) so the residue loop doesn't re-convert per pole.
  10. **Residue loop** (`:246-274`): for each retarded pole `pk` and each entry `(i,j)`: skip entries where `pk` is *not* a pole of *that entry's* denominator (detected via `|Q_entry_cdf[i][j](pk)| > 1e-9` → residue vanishes there). Skip when `|Q′| < 1e-30` (higher-multiplicity). Otherwise `C_entries[i][j] = 1j · P_ij(pk) / Q′_ij(pk)`. Threshold noise below `1e-12 · max|C|`, append `matrix(CDF, C_np)`.
  11. **D_delta** (`:276-295`): leading-coefficient ratio per entry (`deg P == deg Q ⟹ lc(P)/lc(Q)`, else 0), assembled as `matrix(SR, D_delta_data)`.
  12. Return `(pole_vals, C_mats, D_delta)`. On `_PolynomialPathTimeout` or any other exception, log (if verbose) and return `(None, None, None)`.

#### `_trunc_str(s, maxlen=400)` — `:316`

Truncate a string to `maxlen` with an ellipsis. Display helper.

#### `_print_matrix(label, M, resp_names, phys_names)` — `:321`

Print the nonzero entries of an SR matrix with response/physical row/col labels. Verbose-mode display only.

#### `_print_propagator_symbolic_stages(prop, resp_names, phys_names)` — `:335`

Print `D_omega`, `G_ft`, `adj_ft`, `D_delta` (whichever are non-None). Display only.

#### `_print_propagator_stages(prop)` — `:351`

Print everything in a (possibly cache-loaded) propagator dict; derives resp/phys names from `ring_gen_names + nf`. Display only.

#### `factor_propagator(prop, *, per_entry_timeout=5.0, slow_entry_threshold=0.5, verbose=False)` — `:366`

- **Takes:** a propagator dict, per-entry factor timeout, a one-shot "bail" threshold, verbose flag.
- **Returns:** a **new** dict with `G_ft` replaced by a cosmetically factored copy; input untouched.
- **Steps:** apply `_safe_factor` entrywise to `G_ft` via `_factor_with_bail` — a closure that (a) skips if a one-shot `bail` flag is set, (b) times each `factor()`, and (c) flips `bail` permanently after any single entry exceeds `slow_entry_threshold` seconds. This caps total cost to ≈ one slow entry even when the alarm fails to interrupt Singular. Purely cosmetic — for report display; the integrator does not need factored form.

#### `_safe_factor(e, timeout=5.0)` — `:422`

- **Takes:** an SR expression and a timeout.
- **Returns:** `e.factor()` if it completes in time; otherwise the *unfactored* `e`.
- **Steps:** arm `cysignals.alarm.alarm(timeout)`, call `e.factor()`, cancel the alarm in `finally`. Catches the usual symbolic-error tuple, `AlarmInterrupt`, and `SignalError` (by class-name check, because in some Sage builds it isn't an `Exception` subclass). Comment notes the `-n·v` term in spike-reset models enriches the inverse and makes Singular loop while spewing Flint divide-by-zero warnings.

#### `_omega_inf_limit_fast(expr, omega_var)` — `:460`

- **Takes:** an SR rational expression and the ω symbol.
- **Returns:** `SR(0)` if `deg(num) < deg(den)`; `num_lc/den_lc` if degrees equal; `None` if `deg(num) > deg(den)` *or* the expression isn't a clean polynomial ratio in ω (so the caller can fall back to `sage.limit`).
- **Steps:** extract `.numerator()`/`.denominator()`, compare ω-degrees, return the leading-coefficient ratio. **Why it exists:** calling `sage.limit` (Maxima) on raw cofactor/det entries triggers thousands of Flint divide-by-zero warnings and Singular thrashing; this hand-rolled rational-limit avoids Maxima entirely (`:464-470`).

#### `_to_kernel(c, Dt, delta_D, delta_Dp)` — `:507`

- **Takes:** an SR free-action coefficient `c`, and the three namespace symbols `Dt`, `delta_D` (δ), `delta_Dp` (δ′).
- **Returns:** the kernel-form `c0·δ + c1·δ′`.
- **Steps:** if `c` already contains δ, return as-is. Else `p0 = c|_{Dt→0}` (the constant part) and `rest = (c − p0)|_{Dt→δ_Dp}` (the derivative part), return `p0·δ_D + rest`. (Covered in §2 of The math.)

#### `build_propagator(ft, model, cache_dir_root='saved_theories', use_cache=True, verbose=True, force=False)` — `:524`

**The top-level builder.** Returns the propagator data dict (see Data structures).

- **Takes:** an *expanded* `FieldTheory` `ft`, the `model` dict, cache root, cache flags.
- **Returns:** `prop` dict with keys `K_ker, K_ft, G_ft, adj_ft, D_omega, D_delta, t_var, omega, nf, ring_gen_names, pole_vals(=None), C_mats(=None)` and (if spatial) `G_tx_sym, G_tx, spatial_dim, ...`.
- **Steps:**
  1. **Cache lookup** (`:546-583`): slugify `model['name']` → `prop_tag`; `cache = PipelineCache("saved_theories/<tag>")`. If a cached `propagator.sobj` exists and is not forced, load it. **Two staleness guards:** (a) rebuild if cached `nf != ft._n_tilde` (catches field-list edits without a rename); (b) rebuild if the model is spatial but the cache predates the spatial block (`G_tx_sym is None`). On a valid hit, rebuild the unpicklable spatial `G_tx` callables via `make_g_tx_callables(prop)` and return.
  2. **Build K_data from the (1,1) free action** (`:585-606`): get `S_free = ft.free_action()` and the ring generator names. Split names into `resp_names` (first `_n_tilde`) and `phys_names` (rest). Build `pos_to_row`/`pos_to_col` index maps. Iterate over the monomials of `S_free.dict()`; for each, find which generator indices have positive exponent and map them to a `(row, col)`; accumulate the coefficient into `K_data[row][col]`. `K_mat = matrix(SR, K_data)`.
  3. **Kernel form** (`:611-619`): pull `Dt, delta_D, delta_Dp` from `ns`; apply `_to_kernel` entrywise → `K_ker`.
  4. **Fourier transform** (`:624-645`): create `t_var = SR.var('t')`, `omega = SR.var('omega', latex_name=r'\omega')`. Substitute `delta_D → dirac_delta(t)` and `delta_Dp → diff(dirac_delta(t), t)`, then `fourier_transform(...)` each nonzero entry → `K_ft`. Apply the model's `kernel_ft_image(ns, omega)` hook to substitute kernel symbols' frequency images.
  5. **Complexity gate** (`:681-686`): compute the set of free symbols across `K_ft` (minus ω); set `rich = (nf >= 6) or (len(free_syms) > 20)`. The comment explains the *matrix-size* gate (not symbol count) is the right metric because Sage's cofactor inverse is O(nf!) — spike-reset at nf=8 with only ~18 symbols exposed the old symbol-count gate as wrong.
  6. **Symbolic inverse block** (`:688-753`, only when *not* `rich`): under a 120 s SIGALRM budget (`_SYMBOLIC_INVERSE_BUDGET_SEC`), run `_do_symbolic_inverse_block` which computes `G = K_ft.inverse()`, `D_om = K_ft.det()`, `adj = G·D_om` (reusing the inverse), and a symbolic `D_delta` via `_omega_inf_limit_fast` per entry. On timeout / exception / `SignalError`, set `G_ft = adj_ft = D_omega = D_delta = None` so the numerical stage handles everything. When `rich`, skip symbolic inversion entirely and log the reason.
  7. **Assemble `prop`** (`:765-779`) with `pole_vals = None`, `C_mats = None` (filled later).
  8. **Spatial stage** (`:781-812`, only if `model.get('spatial')`): call `build_spatial_propagator(K_ft, omega, ns, model, resp_names, phys_names)` and merge its output. On failure (Tier-1 closed form inapplicable), record `G_tx = None`, `G_tx_sym = None`, and `spatial_tier1_error`.
  9. **Cache save** (`:817-828`): `cache.save('propagator', prop)` *before* attaching the unpicklable `G_tx` callables.
  10. **Attach spatial callables** (`:830-836`, post-save): if `G_tx_sym` present, `prop['G_tx'] = make_g_tx_callables(prop)`.
  11. Return `prop`.

#### `_horner(coeffs, x)` — `:841`

Horner-rule polynomial evaluation `Σ coeffs[i]·x^i` (ascending-degree coefficient list). Empty list ⇒ `0j`. Used to evaluate the cached entrywise numerator/denominator polynomials fast (`_K_at`, `:1062-1063`).

#### `compute_poles_and_residues(prop, num_params, verbose=True)` — `:851`

**Fills `prop['pole_vals']` and `prop['C_mats']` in place. Three-tier strategy.**

- **Takes:** a `prop` dict (from `build_propagator`) and a `num_params` substitution dict (SR.var → float).
- **Returns:** the mutated `prop`.
- **Tier 1 — exact polynomial fraction-field** (`:888-924`): call `_compute_residues_via_polynomial_fracfield`. On success, set `prop['pole_vals']`, `prop['C_mats']`, and fill `prop['D_delta']` if `build_propagator` left it `None`. Print the poles and residue matrices if verbose, then **return early**.
- **Tier 2/3 — numpy cofactor** (only if Tier-1 returned `None`, `:926-1604`): substitute `num_params` into `K_ft` → `K_ft_num`. Two sub-branches:
  - **Rich / heterogeneous path** (`adj_ft is None or G_ft is None`, `:939-1166`):
    1. Symbolic `det(K_ft_num)`; warn if free symbols beyond ω remain (pole-finding would return 0 roots — usually means a kernel symbol wasn't substituted or a saddle value missing from `num_params`).
    2. Extract `det_num_sr` coefficients in ω, trim trailing near-zero leading coefficients, build `char_poly = PR(coeffs)`, find roots with Sage's `char_poly.roots(CDF)` (PARI/MPS under the hood — "more accurate than numpy.roots for mixed-magnitude coefficients").
    3. Cache `K`'s entrywise numerator/denominator coefficients (`K_num_coeffs`, `K_den_coeffs`) and define `_K_at(ω)` (numeric matrix via Horner) and `_G_at(ω) = inv(_K_at(ω))`.
    4. **Fallback pole-finder** (`:1081-1142`) if `roots_all` is empty: sample `det(K_at(ω))` on a circle of radius 2, `np.polyfit` to degree `4·nf`, root the fit, and `_newton_refine` each candidate on the numerical determinant (accept if `|det| < 1e-6` and `imag > 1e-9`, deduped).
    5. Set `prop['adj_ft'] = None`, `prop['G_ft'] = None`.
    6. `D_delta` via a **two-probe scaling test** (`:1150-1166`): evaluate `_G_at(1e8)` and `_G_at(1e10)`; an entry that's the same at both is a constant (→ `δ`); one that scales like 1/ω → 0.
  - **Legacy symbolic path** (`adj_ft` and `G_ft` both present, `:1167-1259`): determinant-first pole finding (comment warns *not* to use the largest entrywise denominator of `G_ft` — it can be a strict divisor of `num(det K)` and give bogus/missing poles). Falls back to the entrywise-denominator method only if det-coefficient extraction fails. Caches the same entrywise `K` coefficients for `_K_at`.
  - **Common: pre-dedup + Newton refine** (`:1261-1445`): prune to retarded roots (`imag > 1e-9`) and dedup at the ~1e-5 root-finder scale; then **Newton-refine each candidate** on the numerical `det(K(ω))` (central-difference derivative). Two Newton blocks exist — one general (`:1297`, with a `det_scale` reference and detailed reject reasons: `no-converge`, `drifted-to-LHP`, `|det| too large`) and a second, near-identical block gated on `adj_ft is None or G_ft is None` (`:1389`). Acceptance: `|det(K(ω_refined))|/det_scale < 1e-9`, stays in the upper half plane, deduped within 1e-7.
  - **Residues** (`:1447-1576`): in **both** sub-branches, compute residues *numerically* via the cofactor adjugate, never the symbolic `adj_ft`:
    - `_cofactor_adj(M)`: `adj[i,j] = (−1)^{i+j}·det(M with row j, col i removed)` (note the transpose — row `j`, col `i`).
    - `d_prime = (det(K(pk+h)) − det(K(pk−h)))/(2h)` (central difference, `h = 1e-6·(1+|pk|)`).
    - `C_np = 1j · adj(K(pk)) / d_prime`, noise-thresholded, stored as `matrix(CDF, ...)`.
    - The lean-symbolic branch (`:1503-1576`) re-extracts its own `K_ft_num` coefficient cache (`_K_at_lean`) and does the same thing. Comment (`:1503-1518`) explains *why numerical, not symbolic*: Sage stores `adj_ft` as sum-of-rationals whose denominators diverge by 10³–10⁴ near kernel poles, catastrophically cancelling and losing 3–5 digits (measured 11–42% error on quad's residues).
  - Set `prop['pole_vals']`, `prop['C_mats']`, print if verbose, return `prop`.

### `msrjd/integration/time_domain/propagator_td.py`

#### Module constant `USE_SIMPLIFY_FULL_IN_GT = False` — `:132`

A gate (default `False`) for the per-entry `simplify_full()` pass on the smooth matrix in `build_G_t_matrix`. The ~120-line comment block (`:56-131`) documents the trade-off: the pass runs Maxima on every entry (~50–80 µs each) to canonicalize. It is **dead weight** for plain rational-propagator theories (every subset is "analytic-eligible" and the smooth matrix is never read by the SR path), giving measured savings of 19–41% on regression fixtures. It **may pay back** only for `NoiseSourceType` vertices with a smooth kernel, where the SR/`scipy.nquad` fallback fires and collapsing duplicate poles shrinks the JIT graph.

#### `build_G_t_matrix(propagator_data, t_var, num_params=None)` — `:135`

- **Takes:** the propagator dict (needs `pole_vals`, `C_mats`, `G_ft`, `nf`, optionally `D_delta`), the symbolic time variable `t_var`, and an optional `num_params` substitution dict.
- **Returns:** a dict `{'smooth': SR matrix, 'delta': SR matrix, 't_var': t_var}`.
- **Steps:**
  1. Pull `pole_vals`, `C_mats`, `G_ft` from the dict.
  2. `_to_sr_ab(value)` (`:225`): normalize *raw* Python `complex` / Sage `ComplexDoubleElement` into `a + b·I` SR expressions with Python `float` components — **but leave genuine SR expressions alone** (so exact closed-form poles like `i·(1 − √6/10)` keep their algebraic content; casting through `complex()` would silently lose precision). Apply to every pole and residue entry. The comment (`:204-219`) explains this **must** run before `.subs(num_params)` to avoid a GiNaC `TypeError: '<' not supported between instances of 'complex' and 'complex'` when an opaque complex node hits a term-sort comparison later.
  3. If `num_params`, substitute into poles and residue entries.
  4. Determine `nf` from `propagator_data['nf']` (falling back to `C_mats[0].nrows()` or `D_delta.nrows()`, else raise `ValueError`).
  5. If `pole_vals` is empty, **warn** (the smooth part will be identically zero; only `D_delta` contributes — usually means the char-poly solve found no retarded roots).
  6. Build `smooth = Σ_k C_mats[k]·exp(I·pole_vals[k]·t_var)` as a proper `matrix(SR, nf, nf, 0)` accumulator (so it's a matrix even when empty). Optionally `simplify_full()` if `USE_SIMPLIFY_FULL_IN_GT`.
  7. **Delta coefficients:** prefer the precomputed `D_delta` (apply `num_params` if given); otherwise compute per entry via Sage's symbolic `limit(G_ft[i,j], ω→∞)` using `_infer_omega_variable` to find the ω symbol.
  8. Return `{'smooth', 'delta', 't_var'}`.

#### `_infer_omega_variable(G_ft, num_params)` — `:353`

- **Takes:** the `G_ft` matrix and optional `num_params`.
- **Returns:** the unique free variable assumed to be ω (preferring one literally named `'omega'`), or `None` if no free variables remain.
- **Steps:** collect free variables across all entries (after `num_params` subs); prefer `'omega'` by name, else the alphabetically-first.

#### `G_t_entry(G_t_obj, phys_idx, resp_idx, t_expr, include_heaviside=True)` — `:379`

- **Takes:** the `build_G_t_matrix` output (dict or bare SR matrix), physical row index, response col index, the time argument `t_expr` (for an edge `u→v` this is `t_v − t_u`), and a Heaviside flag.
- **Returns:** the SR expression `smooth[phys_idx, resp_idx]` with its internal `t_var` substituted by `t_expr`, optionally `× heaviside(t_expr)`.
- **Steps:** unwrap `'smooth'` from a dict (or use the bare matrix); `_infer_time_variable` to find the time symbol; substitute `t_var → t_expr`; multiply by `heaviside(t_expr)` if requested. **This is the (phys, resp) transpose** of the kernel's (resp, phys) layout — the canonical per-edge propagator lookup used by the tree integrators. *Only the smooth part is returned*; the δ(t) part is handled separately via `G_t_delta_coeff`.

#### `G_t_delta_coeff(G_t_obj, phys_idx, resp_idx)` — `:433`

- **Takes:** the `build_G_t_matrix` output and the (phys, resp) indices.
- **Returns:** the δ(t) coefficient `lim_{ω→∞} Ĝ[phys, resp](ω)` as a Python `complex` (or a real `float` if the imaginary part is negligible). Returns `0.0+0.0j` if there's no δ component or the input is a bare smooth matrix.
- **Steps:** read `delta[phys_idx, resp_idx]`, coerce via `complex(CDF(val))`, and demote to a real `float` if `|imag| < 1e-12·max(|real|,1)`.

#### `_infer_time_variable(G_t_matrix)` — `:459`

- **Takes:** the smooth SR matrix.
- **Returns:** the time variable `t` (the only remaining free variable once parameters are substituted), or `None`.
- **Steps:** scan entries; return the first entry's first free variable. **Avoids `entry == 0`** equality (which routes to Maxima and *fails* on embedded Python `complex` coefficients — `TypeError` comparing complexes); checks `entry.variables()` directly and only uses `is_trivial_zero()` for scalar entries.

---

## Data structures

### The propagator dict `prop` (the central object)

Produced by `build_propagator`, mutated by `compute_poles_and_residues`, consumed by `build_G_t_matrix`. Keys:

| Key | Type | Meaning |
|-----|------|---------|
| `K_ker` | `matrix(SR)` `nf×nf` | Kernel-form kinetic matrix (`c0·δ + c1·δ′`), layout [resp, phys]. |
| `K_ft` | `matrix(SR)` `nf×nf` | Fourier image of `K_ker`; entries are rational in ω (plus parameters/kernel symbols). |
| `G_ft` | `matrix(SR)` or `None` | Symbolic propagator `K_ft⁻¹` (None for "rich" models or budget-busts). |
| `adj_ft` | `matrix(SR)` or `None` | Adjugate `G_ft·det(K_ft)` (None when `G_ft` is None). |
| `D_omega` | SR or `None` | `det(K_ft)`; its numerator's roots are the poles. |
| `D_delta` | `matrix(SR)` or `None` | Instantaneous limit `lim_{ω→∞} G_ft(ω)` — δ(t) coefficients. |
| `t_var` | SR var | The symbolic `t`. |
| `omega` | SR var | The symbolic ω (`latex_name=\omega`). |
| `nf` | int | Number of physical (= response) fields; matrix dimension. |
| `ring_gen_names` | list[str] | All `2·nf` generator names; first `nf` = response, rest = physical. |
| `pole_vals` | list[complex] or `None` | Retarded poles ω_k (Im > 0), sorted by `(imag, real)`. Filled by `compute_poles_and_residues`. |
| `C_mats` | list[`matrix(CDF)`] or `None` | One `nf×nf` residue matrix per pole; `C_k = i·Res G(ω_k)`. Filled by `compute_poles_and_residues`. |
| `G_tx_sym`, `G_tx`, `spatial_dim`, `spatial_tier1_error`, … | — | Only present for spatial models (heat-kernel block). |

### The `build_G_t_matrix` output dict

| Key | Type | Meaning |
|-----|------|---------|
| `smooth` | `matrix(SR)` `nf×nf` | `Σ_k C_mats[k]·exp(I·p_k·t)` — the analytic pole-residue part. Caller multiplies by `Θ(t)`. |
| `delta` | `matrix(SR)` `nf×nf` | `lim_{ω→∞} Ĝ[i,j]` — δ(t) coefficients (mostly zero). |
| `t_var` | SR var | The time variable used to build `smooth`. |

### Tuple returned by `_compute_residues_via_polynomial_fracfield`

`(pole_vals: list[complex] | None, C_mats: list[matrix(CDF)] | None, D_delta: matrix(SR) | None)`.

### Exact-arithmetic scratch objects (inside the polynomial path)

- `CF = CyclotomicField(4)` — the field ℚ(i).
- `PR_exact = CF[ω]`, `F_exact = Frac(CF[ω])` — exact polynomials / rational functions.
- `Q_poly ∈ CF[ω]` — universal denominator (LCM of per-entry denominators).
- `P_cdf[i][j]`, `Q_per_entry[i][j]`, `Q_entry_cdf[i][j]`, `Q_entry_prime_cdf[i][j]` — `CDF[ω]` numerator/denominator caches for the residue loop.

### Entrywise numeric polynomial caches (numpy paths)

`K_num_coeffs[i][j]`, `K_den_coeffs[i][j]` — ascending-degree lists of Python `complex` coefficients so `_K_at(ω)` evaluates `K(ω)` as a numpy complex matrix in microseconds via Horner.

---

## Data flow

```
build_propagator INPUTS:
    ft     : expanded FieldTheory (provides ft.ring(), ft.free_action(),
             ft._n_tilde, ft._ns with Dt/delta_D/delta_Dp/kernel symbols)
    model  : dict with keys
               'name'            → cache slug
               'kernel_ft_image' → callable(ns, omega) → subs dict g→ĝ(ω)
               'spatial'         → optional spatial-field declaration
   ▼
build_propagator OUTPUT:
    prop dict (see Data structures); pole_vals/C_mats = None.
    Side effect: writes saved_theories/<slug>/propagator.sobj

compute_poles_and_residues INPUTS:
    prop       : the dict above
    num_params : dict {SR.var: float}  (from the mean-field solve)
   ▼
compute_poles_and_residues OUTPUT (mutation):
    prop['pole_vals'] : list[complex]   (Im>0, sorted)
    prop['C_mats']    : list[matrix(CDF)]
    prop['D_delta']   : filled if it was None

build_G_t_matrix INPUTS:
    propagator_data = prop, t_var, num_params
   ▼
build_G_t_matrix OUTPUT:
    {'smooth', 'delta', 't_var'}
   ▼
G_t_entry(G_t_obj, phys_idx=j, resp_idx=i, t_expr=t_v−t_u) → SR
G_t_delta_coeff(G_t_obj, j, i) → complex
   ▼  (consumed in final_integral.py / grouped_integral.py)
    smooth_factor = G_t_entry(...)            (final_integral.py:2837)
    delta_c       = G_t_delta_coeff(...)      (final_integral.py:2836)
```

**Concrete OU-like example.** For `S_free ⊃ ñ(∂_t n + μ n)`: `K_ker = [[0, iω... actually]]` — more precisely `K[ñ, n] = δ′ + μ·δ`, so `K_ft[ñ, n] = iω + μ`. With one field (`nf=1`), `G_ft = 1/(iω + μ)`, the single pole is `ω = iμ` (Im > 0 for μ > 0), the residue is `1/i = −i` so `C_0 = i·(−i) = 1` (giving `G(t) = Θ(t)·e^{−μt}`), and `D_delta = 0` (proper). A two-field `(n, v)` system (e.g. spike-reset) produces a 2×2 `K_ft` and 2 poles, with the `ñ ↔ δn` coupling yielding a nonzero `D_delta` entry → a `δ(t)` in `G_R`.

---

## Gotchas & caveats

1. **`msrjd/core/propagator.py` is a stub, not the implementation.** Despite the name, it contains only a docstring ("Not yet extracted — logic currently lives in the demo notebook"). The real code is `pipeline/_propagator.py`. Do not look for `build_propagator` in `msrjd/core/propagator.py`.

2. **Retarded = Im(ω) > 0 is a hard convention, born from `e^{−iωt}`.** Every pole filter uses `imag > 1e-9`. If you ever change `fourier_transform` to `e^{+iωt}`, every pole-side sign flips and the entire downstream causality machinery breaks. (`propagator_td.py:12-20`, `_propagator.py:1261-1266`.)

3. **The symbolic inverse is deliberately skipped for "rich" matrices.** Sage's `K_ft.inverse()` is cofactor-based, O(nf!). For `nf ≥ 6` (or >20 free symbols) `build_propagator` skips symbolic inversion entirely and leaves `G_ft = adj_ft = D_omega = D_delta = None`, deferring everything to the numerical stage (`:681-686`, `:754-763`). The gate switched from symbol-count to matrix-size because spike-reset (nf=8, ~18 symbols) ran for *hours* despite a lean symbol set.

4. **`adjugate()` is computed as `G·det`, never via Sage's `.adjugate()`.** `K_ft.adjugate()` re-expands all n² cofactor sub-determinants without reusing the inverse — measured **734 s vs ~11 s** on a 4×4 spike-reset matrix (`:710-714`).

5. **`sage.limit` is avoided for ω→∞.** It routes through Maxima and triggers thousands of Flint divide-by-zero warnings + Singular thrashing on raw cofactor entries. `_omega_inf_limit_fast` does the leading-coefficient ratio by hand (`:464-470`).

6. **Residues are ALWAYS computed numerically, even when a symbolic `adj_ft` exists.** Sage stores `adj_ft` as sum-of-rationals whose per-term denominators diverge by 10³–10⁴ near kernel poles, catastrophically cancelling and losing 3–5 digits (measured **11–42% relative error** on quad's residues). The numpy cofactor path has no such leak (`:1503-1518`).

7. **Non-squarefree denominators (multiplicity ≥ 2 poles) are NOT handled.** The single-pole residue formula `i·P/Q′` can't represent the `(τ·e^{−pt})` Jordan-block term. The polynomial path *bails* on `not Q_poly.is_squarefree()` (`:196-209`); the numpy path *silently leaves a zero residue* when `|Q′| < 1e-30` or `|d_prime| < 1e-30` (`:261-266`, `:1490-1495`). A "v2" would need the m-th derivative residue plus a downstream integrator that can absorb `τ^k·e^{−pt}` modes. This is the most significant *known limitation* of the subsystem.

8. **`_compute_residues_via_polynomial_fracfield` requires QQ-rationalizable parameters.** `_sr_complex_to_CF` calls `QQ(c.real_part())`; a non-rationalizable (e.g. irrational symbolic) parameter throws and the whole tier returns `(None,None,None)`, falling to numpy cofactor (`:122-126`).

9. **Universal denominator must be an LCM, not the first entry's denominator.** Upper-triangular `K_ft` from the Markovian-embedding preprocessor gives entries with *different* canonical denominators; picking the first under-counts poles. The code uses `Q_poly = LCM(per-entry denominators)` (`:172-181`).

10. **Pole-finding silently returns 0 roots if free symbols beyond ω remain.** If a kernel symbol wasn't substituted (`kernel_ft_image` hook missing an entry) or a saddle value isn't in `num_params`, `det(K_ft_num)` has non-numeric coefficients and the root-finder returns nothing. The rich path *warns* about this explicitly (`:976-982`); the empty-`pole_vals` case then surfaces as a warning in `build_G_t_matrix` (`:271-283`).

11. **`factor()` can hang or segfault Singular.** `build_propagator` returns the *raw* (unfactored) inverse on purpose. `factor_propagator`/`_safe_factor` apply factoring only cosmetically for reports, with a cysignals alarm *and* a one-shot bail flag because the alarm "does not always interrupt Singular's native loop" (`:376-391`, `:441`).

12. **The cysignals `SignalError` is caught by class name, not isinstance.** In some Sage builds it isn't a subclass of `Exception`, so `_safe_factor` and `build_propagator` catch `BaseException` and check `exc.__class__.__name__ == 'SignalError'`, re-raising everything else (so `KeyboardInterrupt`/`SystemExit` aren't swallowed) (`:451-457`, `:746-753`).

13. **`_to_sr_ab` must run before `.subs(num_params)`.** Otherwise `SR(python_complex)` becomes an opaque GiNaC node that later raises `TypeError: '<' not supported between instances of 'complex' and 'complex'` during a term-sort comparison (`propagator_td.py:204-219`).

14. **Genuine SR poles are preserved exactly; only raw complex/CDF are normalized.** `_to_sr_ab` deliberately does *not* cast SR expressions through `complex()` — that would collapse exact closed-form poles to double precision and bias correlators across many τ evaluations (`propagator_td.py:230-233`).

15. **`Θ(0)` semantics differ by path.** The numerical integrator uses `Θ(0) = 0` (Δt = 0 strictly excluded from retarded support, enforced by the Heaviside-filter integrand and strict polytope inequalities). But Sage's symbolic `heaviside(0)` returns `1/2` — used *only* in the symbolic-display path (`propagator_td.py:30-43`).

16. **Two nearly-identical Newton-refine blocks exist in `compute_poles_and_residues`.** One general block (`:1297`) and a second gated on `adj_ft is None or G_ft is None` (`:1389`). For the rich/heterogeneous path, candidates appear to be Newton-refined *twice* (once by each block). This is harmless (the second refine of an already-converged root is a near-no-op) but is duplicated logic worth flagging (see *open_questions*).

17. **Cache is keyed by model name slug, taylor-order-independent.** `build_propagator` caches to `saved_theories/<slug>/propagator.sobj`. The cache auto-invalidates if cached `nf != ft._n_tilde` (field-list edits) or if a spatial model's cache predates the spatial block. Renaming the model's *parameters* without renaming the *model* will NOT invalidate the cache — but since `compute_poles_and_residues` re-substitutes `num_params` fresh each run, the cached symbolic stages are still correct.

18. **`cofactor_adj` uses a transposed index convention.** `adj[i,j] = (−1)^{i+j}·det(M with row j, col i removed)` — note row `j`, col `i` (the adjugate is the transpose of the cofactor matrix). Getting this backwards would transpose every residue (`:1470-1479`).

---

## Glossary

- **MSR-JD field theory** — Martin–Siggia–Rose–Janssen–De Dominicis: a path-integral formulation of stochastic dynamics where each physical field `φ` gets a conjugate *response* field `φ̃`. The free action is bilinear `φ̃·K·φ`.
- **Response / physical / tilde field** — `φ̃ᵢ` (response, "tilde") and `φⱼ` (physical). The kernel matrix `K` couples one of each.
- **Free action / bilinear (1,1) sector** — the Gaussian part of the action: exactly one response leg and one physical leg. `ft.free_action()` returns the bigrade-(1,1) polynomial.
- **Kernel matrix K** — the matrix of bilinear coefficients, layout `[resp_row, phys_col]`.
- **Propagator G = K⁻¹** — the two-point Gaussian correlation/response function; the building block of every Feynman edge.
- **`K_ker` (kernel form)** — `K` written with abstract `δ`/`δ′` symbols (`c0·δ + c1·δ′`) so it Fourier-transforms cleanly.
- **`K_ft` / `G_ft`** — Fourier images (functions of ω): `G_ft = K_ft⁻¹`.
- **Adjugate `adj(K)`** — the transpose of the cofactor matrix; `K⁻¹ = adj(K)/det(K)` (Cramer's rule).
- **Characteristic polynomial / `D_omega`** — `det(K_ft)`; its numerator's roots are the system poles.
- **Pole ω_k** — a (retarded) zero of `det(K_ft)` with Im(ω_k) > 0; sets a relaxation/oscillation rate.
- **Residue matrix `C_mats[k]`** — `i·Res_{ω_k} G_ft`; the amplitude matrix of mode `k`.
- **`D_delta` (instantaneous limit)** — `lim_{ω→∞} G_ft(ω)`; the δ(t) coefficient matrix of the time-domain propagator.
- **Retarded propagator `G_R(t)`** — the causal propagator: `D_delta·δ(t) + Θ(t)·Σ_k C_k·e^{i·ω_k·t}`.
- **Heaviside `Θ(t)`** — the step function enforcing retardation (response only after cause).
- **Proper / strictly proper rational** — `deg(num) < deg(den)`; vanishes at ∞ (no δ term). "Non-proper" has a polynomial part → δ(t).
- **`kernel_ft_image` hook** — a model-supplied callable `(ns, omega) → subs-dict` mapping abstract kernel symbols `g` to their frequency images `ĝ(ω)`.
- **SageMath / `SR`** — the CAS and its Symbolic Ring (GiNaC/Pynac-backed). See External tools.
- **`CDF` / `QQ`** — Sage's Complex Double Field (machine-precision complex) and rational field ℚ.
- **`CyclotomicField(4)` = ℚ(i)** — rationals adjoined the imaginary unit; gives exact, reliable polynomial GCD.
- **`PolynomialRing` / fraction field** — `R[ω]` and `Frac(R[ω])`; the latter reduces each entry to canonical irreducible `P/Q`.
- **Horner's rule** — efficient nested polynomial evaluation `((c_n·x + c_{n−1})·x + …)`.
- **Cofactor expansion** — computing a determinant/adjugate via minors (the dense numpy path uses `np.delete` to form minors).
- **Newton refinement** — Newton's method on `det(K(ω))` to polish a root to machine precision.
- **Catastrophic cancellation** — loss of significant digits when subtracting nearly-equal large quantities (the reason symbolic `adj` residues are unreliable near poles).
- **Squarefree polynomial** — no repeated roots; a non-squarefree `Q` signals a multiplicity-≥2 (Jordan-block) pole the single-pole formula can't handle.
- **cysignals `alarm` / `SignalError` / `AlarmInterrupt`** — Cython tools to time-bound and recover from native (Singular/Pynac) routines.
- **SIGALRM** — POSIX alarm signal; the `signal`-module wall-clock budget mechanism (absent on Windows).
- **`.sobj`** — a Sage-pickled object file; the propagator cache format.
- **`analytic-eligible` / `NoiseSourceType`** — downstream integrator notions: a subset is analytic-eligible (closed-form, no scipy.nquad) iff there's no smooth-kernel noise vertex; relevant to whether `simplify_full` pays off.

---

## Proposed manual subsections

1. **Role of the propagator in the MSR-JD pipeline** — why `G = K⁻¹` gates everything; what feeds and consumes it.
2. **From action to kernel matrix `K`** — the (1,1) bilinear sector, layout [resp, phys], the `K_data` extraction loop.
3. **Kernel form and the Fourier convention** — `δ`/`δ′` symbols, `_to_kernel`, `e^{−iωt}` and why poles live in Im(ω) > 0.
4. **The frequency-domain propagator** — `G_ft = K_ft⁻¹`, adjugate, determinant; the budget-gated symbolic inverse and the rich/lean split.
5. **Poles as zeros of det K_ft** — the characteristic polynomial, retarded filtering, Newton refinement, spurious-root rejection.
6. **Residues: three tiers** — exact ℚ(i)[ω] fraction-field, numpy cofactor adjugate, legacy symbolic; the `C_k = i·Res` convention; why residues are always numerical.
7. **The instantaneous limit `D_delta`** — proper vs non-proper rationals, leading-coefficient ratio, the two-probe scaling test.
8. **The time-domain propagator `G_R(t)`** — `D_delta·δ(t) + Θ(t)·Σ C_k e^{iω_k t}`, `build_G_t_matrix`, the (phys, resp) transpose, per-edge `G_t_entry`/`G_t_delta_coeff`.
9. **Robustness machinery** — SIGALRM/cysignals budgets, the `factor()` hazard, the GiNaC complex-comparison pitfall, `Θ(0)` semantics.
10. **Caching** — `PipelineCache`, `.sobj`, staleness guards, taylor-order independence, unpicklable spatial callables.
11. **Known limitations and failure modes** — multiplicity-≥2 poles, non-rationalizable parameters, free-symbols-beyond-ω, empty pole lists.

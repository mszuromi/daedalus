# Spatial Coupled Fields, Dyson Dressing, Operator IR & Bridge

> Subsystem slug: `spatial-coupled`
> Primary files:
> - `msrjd/integration/spatial/spectral_propagator.py`
> - `msrjd/integration/spatial/dyson_dressing.py`
> - `msrjd/integration/spatial/loop_dyson.py` (ORACLE-ONLY — off the production path)
> - `msrjd/integration/spatial/pipeline_bridge.py` (2157 lines — the spatial→shared-pipeline bridge)
> - `pipeline/spatial_operator_ir.py`

---

## Overview

A "spatial" MSR-JD theory is a stochastic field theory whose fields live on space **and** time, e.g. a reaction–diffusion equation `∂_t φ = -μ φ + D ∇²φ + (nonlinear) + ξ`. The framework's job is to compute correlation functions `C(x, τ) = ⟨φ(x₀+x, t₀+τ) φ(x₀, t₀)⟩` (and higher cumulants) perturbatively, diagram by diagram. This subsystem is the part of that machine that handles three things the simpler "diagonal single-field" path cannot:

1. **Coupled multi-field theories.** When there are several fields `φ₁…φ_N` whose linear dynamics mix (an off-diagonal reaction/"mass" matrix `M`), the free propagator is a matrix exponential `e^{-Mt}`, not a product of independent scalar modes. `spectral_propagator.py` builds this via the spectral (eigenprojector) decomposition of `M`.

2. **Unequal diffusion.** When the diffusion matrix `𝒟` is not a scalar multiple of the identity (`𝒟 ≠ D₀·I`), `M` and `𝒟` do not commute, so there is no clean closed-form propagator. `dyson_dressing.py` implements the **Dyson–Duhamel series**: it splits `𝒟 = D₀·I + 𝒟̂`, treats the residual `𝒟̂|k|²` as a perturbation, and re-sums the correction order by order.

3. **Derivative-vertex theories via an operator IR.** A vertex like Model B's conserved `∇²(φ²)`, Burgers' `∂_x(φ²)`, or KPZ's `(∂_x φ)²` carries spatial derivatives. `pipeline/spatial_operator_ir.py` hosts these as inert symbolic operator nodes (`Lap`, `Dt`, `Dx`), provides the operator algebra (linearity, saddle expansion, mean annihilation), and lowers them to ordinary ring generators so the existing Taylor-expansion machinery works unchanged. Each derivative vertex deposits a **momentum form factor** that the loop integrator then averages.

**Where it sits in the pipeline.** The end-to-end flow is:

```
theory file (.theory.py)
  → SpatialTheoryBuilder / theory_compiler  (operator IR lowering lives here)
  → FieldTheory.expand()                    (multivariate Taylor → vertices, sources)
  → build_propagator / build_spatial_propagator  (the `prop` dict; ac_mass/ac_diffusion or K_ft)
  → compute_cumulants  (the public entry; dispatches by spatial_dim/coupling)
      → pipeline_bridge.compute_spatial_correlator_via_pipeline   (tree, diagonal)
      → pipeline_bridge.compute_coupled_tree_correlator           (tree, coupled)
      → pipeline_bridge.compute_spatial_correlator_generic        (loops, single-field)
      → pipeline_bridge.compute_coupled_loop_correlator           (loops, coupled)
      → pipeline_bridge.compute_spatial_kpoint / compute_coupled_kpoint  (k≥3)
  → C(x, τ) array  →  notebook / sim comparison
```

`pipeline_bridge.py` is the dispatcher and assembler. It **feeds on** the `FieldTheory` object (`ft`), the model dict (`model`), the propagator dict (`prop`), and numeric parameters (`num_params`). It **calls into** the shared diagram pipeline (enumeration, typing, classification, the time-domain `compute_correction_td`), the spatial `full_integrator` (the one genuine loop integral), `spectral_propagator`/`dyson_dressing` (coupled tree), and the analytic q→x Fourier transforms in `spatial_correlator`. It **produces** the real-space correlator array `C[len(tau_grid), len(spatial_grid)]` plus an `info` dict.

The defining design principle (stated in the `pipeline_bridge` module docstring) is **"route through the shared pipeline, then certify."** Rather than reimplementing diagrammatics for the spatial case, the bridge substitutes `Laplacian → -q²` and runs the *same* enumeration/classification/Phase-J path a time-only theory uses; it then *certifies* that the diagram answer `C(q, τ)` equals the per-mode heat-kernel structure read off the propagator, before doing the external `q → x` Fourier transform analytically.

---

## The math

### 1. The linear (free) coupled propagator

For an `N`-component field the linearized inverse propagator (the bilinear "kernel" in MSR-JD) is, in the mixed frequency/momentum representation `(ω, k)`:

```
K(ω, k) = -iω·I + M + 𝒟·|k|²
```

- `M` = reaction (mass) matrix = `diag(μ_i) − A⁽⁰⁾`. It need **not** be diagonal; off-diagonal entries are linear couplings between fields.
- `𝒟` = diffusion matrix = `diag(D_i) + A⁽²⁾`. It need **not** be proportional to `I`.

The retarded (free) propagator is the matrix Green's function `G_R(t,k) = Θ(t)·e^{-(M + 𝒟|k|²)t}`. The trouble: if `M` and `𝒟` do not commute, `e^{-(M+𝒟|k|²)t}` is not a product of two matrix exponentials and has no simple `k`-dependence.

**Reference-diffusion split.** Write
```
𝒟 = D₀·I + 𝒟̂,    D₀ ∈ ℝ,    𝒟̂ = residual (= 0 iff 𝒟 ∝ I).
```
The reference kernel `K₀ = -iω·I + M + D₀|k|²·I` has *scalar* diffusion, which commutes with everything. Diagonalize `M` by its spectral projectors:
```
M = Σ_α m_α P_α,    Σ_α P_α = I,    P_α P_β = δ_αβ P_α.
```
(`m_α` are the eigenvalues of `M`; `P_α` is the rank-1 projector onto the α-th eigenspace, built as the outer product of the right eigenvector with the corresponding row of the inverse eigenvector matrix.) Then the **reference propagator** is closed form (paper Appendix B eq. B23):
```
G₀(t, k) = Θ(t)·Σ_α P_α·e^{-(m_α + D₀|k|²)t}  =  Θ(t)·e^{-Mt}·e^{-D₀|k|²t}.
```

Two exactness facts:
- **𝒟̂ = 0** (scalar diffusion, even with coupled `M`): `G₀` is the **exact** full propagator. No series needed. This already unlocks coupled-reaction / equal-diffusion theories.
- **`M` and `𝒟` both diagonal**: `G₀` reduces to the per-field scalar heat kernel `e^{-(μ_i + D_i|k|²)t}` that the diagonal pipeline (`heat_kernel.py`) already builds.

Only **diagonalizable** `M` is handled; a defective `M` (repeated eigenvalues with non-trivial Jordan blocks) would need the resolvent/confluent form and is deferred. The code guards this with an eigenvector-conditioning cap (`_COND_CAP = 1e10`).

### 2. The tree-level coupled 2-point: Lyapunov / fluctuation–regression

The free 2-point of a coupled *linear* theory is **not** a sum of independent OU modes. With the relaxation matrix `A(q) = M + 𝒟|q|²` and the symmetric noise covariance `N` (read from the `(2,0)` response-field sector, `⟨ξξᵀ⟩ = N`), the stationary covariance solves the continuous Lyapunov equation and the 2-point follows by the fluctuation–regression theorem:
```
A(q) Σ(q) + Σ(q) A(q)ᵀ = N           (stationary covariance Σ)
C(q, τ) = e^{-A(q)|τ|} Σ(q)   (τ ≥ 0),
C(q, -τ) = Σ(q) e^{-A(q)ᵀ|τ|}  (i.e. C(q,τ) = C(q,|τ|)ᵀ).
```
For scalar diffusion `A(q)=M+D₀|q|²·I` shares `M`'s eigenprojectors, so `e^{-A|τ|}` *is* `G₀(|q|²,|τ|)` and this is **exact**. Unequal diffusion needs the Dyson series.

### 3. The Dyson–Duhamel series (unequal diffusion)

With `𝒟 = D₀·I + 𝒟̂` and `𝒟̂ ≠ 0`, the retarded propagator expands in powers of the insertion `𝒟̂|k|²` (paper Appendix B §B.24–B.30):
```
G_R(t,k) = Σ_{n≥0} G_n(t,k),
G_n(t,k) = (-|k|²)^n · e^{-D₀|k|²t} · 𝓗_n(t).                  (B26)
```
The matrix factor `𝓗_n(t)` comes from the n-fold **Duhamel convolution** (each `𝒟̂` insertion is sandwiched between reference evolutions):
```
G_n(t) = (-|k|²)^n ∫_{t ≥ s_1 ≥ … ≥ s_n ≥ 0}
           e^{-M(t-s_1)} 𝒟̂ e^{-M(s_1-s_2)} 𝒟̂ ⋯ e^{-M s_n} d𝐬.
```
Inserting `M = Σ m_α P_α` turns this nested time integral into a **divided difference**. With `f(z) = e^{-zt}`, the nested simplex integral is (up to sign) the n-th divided difference `f[m_{α_0},…,m_{α_n}]`, and shift-invariance factors out `e^{-m_{α_0}t}`:
```
𝓗_n(t) = Σ_{α_0…α_n} P_{α_0} 𝒟̂ P_{α_1} ⋯ 𝒟̂ P_{α_n} ·
          e^{-m_{α_0}t} · Φ_n(t; m_{α_1}-m_{α_0}, …, m_{α_n}-m_{α_0}).   (B27)
```

The one genuinely new primitive is `Φ_n`, the nested-simplex time integral
```
Φ_n(t; ν_1,…,ν_n) = ∫_{σ_n} tⁿ · e^{-t·Σ_i u_i ν_i} d𝐮,    Φ_0(t) = 1,
σ_n = { u_i ≥ 0, Σ u_i ≤ 1 }  (the standard simplex).
```
By the **Hermite–Genocchi formula** with `f(z)=e^{-tz}` (so `f⁽ⁿ⁾(z) = (-t)ⁿ e^{-tz}`) and nodes `{0, ν_1,…,ν_n}`:
```
Φ_n(t; ν) = (-1)ⁿ · f[0, ν_1, …, ν_n]   (the n-th divided difference of f).
```

Worked `n=1` check: with `Z = [[0,1],[0,ν]]`, `expm(-tZ)[0,1] = -t·(1-e^{-tν})/(tν) = -(1-e^{-tν})/ν`; times `(-1)` gives `Φ_1 = (1-e^{-tν})/ν = ∫₀¹ t·e^{-tuν} du`. ✓

### 4. Confluent-safe evaluation: the Opitz theorem

A divided difference written naively as `Σ f(x_i)/∏(x_i - x_j)` blows up (0/0) when two nodes coincide or nearly coincide — and the eigenvalue differences `ν = m_{α_i} - m_{α_0}` routinely repeat (e.g. when several `m_α` are equal, or come as complex-conjugate pairs of a real `M`). The **Opitz theorem** evaluates a divided difference without any division by node differences:

> For analytic `f`, build the `(n+1)×(n+1)` upper-bidiagonal matrix `Z` with the nodes `{0, ν_1,…,ν_n}` on the diagonal and ones on the superdiagonal. Then `f(Z)[0, n] = f[0, ν_1, …, ν_n]` (the top-right corner of `f` applied to `Z`).

So `Φ_n(t; ν) = (-1)ⁿ · expm(-t·Z)[0, n]`, computed by a single matrix exponential of a tiny `(n+1)×(n+1)` matrix. This is exact at coincident nodes and numerically stable at near-coincident ones (no cancellation catastrophe). For the practical truncation orders `n ≤ ~4` the matrices are tiny, so correctness is favoured over speed.

### 5. Dressed tree 2-point (q-space)

With the dressed retarded propagator in hand, the dressed tree 2-point is
```
C^{(N)}(q, τ ≥ 0) = ∫₀^∞ ds  G_R^{(N)}(τ+s, q) · N · G_R^{(N)}(s, q)ᵀ,
```
evaluated by Gauss–Legendre quadrature on a `σ`-concentrated substitution `s = cap·v²` (nodes packed near `s=0`, where the integrand is largest; `cap` set by the slowest eigenvalue). Convergence in the truncation order `N` requires the **residual ratio** `‖𝒟̂‖/D₀ < 1` (geometric in `N` at large `q`; exact at `q=0` where the insertion vanishes).

### 6. Loop-level coupled fields: spectral assignments

For loops with scalar diffusion (`𝒟̂=0`), every edge's *momentum* factor is the same heat kernel `e^{-D₀k²w}` — so the Symanzik (parametric-momentum) machinery is untouched. The coupling lives entirely in the time/matrix factor: each retarded segment carries `Σ_α P_α e^{-m_α w}`. Expanding every segment (an `R` edge = one segment; a `C`/correlation edge = two glued half-segments) turns one coupled diagram into a **weighted sum of scalar diagrams**:
```
value(Γ) = pv · Σ_{{α_r}} ∏_r [P_{α_r}]_{p_r r_r} · I_spec({m_{α_r}})
```
where `(r_r, p_r)` are the segment's (response, physical) matrix indices threaded through `CEdge.fpairs`, `pv` is the enumeration `𝒮(Γ)·prefactor`, and `I_spec` is the shared Symanzik/chamber quadrature evaluated for one mass assignment (all assignments batched against one integral pass). Loop-level **unequal** diffusion layers the Dyson `𝓗_n` insertions onto individual segments via a generalized partial-fraction expansion of the n-fold Duhamel convolution (the `_pf_labels`/`_hn_labels` machinery, log-derivative Bell expansion for repeated masses).

### 7. The operator IR and form factors

A spatial differential operator on a leg of momentum `k`, frequency `ω`, has a simple Fourier image:
```
∇² (Lap) → -|k|²,    ∂_t (Dt) → -iω,    ∂_{x_i} (Dx_i) → i k_i,
```
composed multiplicatively along an operator chain (`∇⁴ = Lap∘Lap → |k|⁴`). A derivative *bilinear* term (degree ≤ 2 in the fields) folds into the propagator kernel `K(ω,k)`; a derivative *vertex* (degree ≥ 3) instead deposits a per-leg **form factor** `Rcal(q, ℓ)` on the diagram, a product over the interaction vertices:
```
Rcal(q, ℓ) = ∏_{interaction vertices v} 𝔉(v),
𝔉(v) = Σ_{type t : n_phys(t)=deg(v)} w_t · 𝔣_t(v),
```
where `w_t = c_t / Σc` is the coupling weight, and `𝔣_t` is the type's Fourier kernel evaluated either at the response-leg momentum (`'composite'` mode: `∇²(φ²)`, `∂_x(φ²)`) or as a product over physical-leg momenta (`'perleg'` mode: `(∂_xφ)²`). The loop integrator averages this polynomial form factor over the loop-momentum Gaussian, either by Gauss–Hermite quadrature (numerical) or — the principled route — by a closed-form **joint `(ℓ,q)`-Gaussian moment** (Wick/Isserlis), which is exact and needs no q-grid.

---

## External tools used

This subsystem touches several third-party libraries. Each is explained from scratch below, with the actual import lines and call sites.

### NumPy

**What it is.** NumPy is Python's foundational numerical-array library. Its central object is the `ndarray`, a dense multi-dimensional array of a fixed numeric dtype (`float`, `complex128`, `int`). NumPy provides vectorized elementwise arithmetic, linear algebra (`np.linalg`), and broadcasting (operations between arrays of compatible-but-different shapes). It is fast because the inner loops run in compiled C and *release the GIL* (the global interpreter lock) during heavy array operations — which is why this subsystem's only safe parallelism is *thread*-based (see Gotchas).

**How this code uses it.** Pervasively. Examples:
- `import numpy as np` (every file).
- `np.linalg.eig(M)` and `np.linalg.inv(V)` in `spectral_projectors` (`spectral_propagator.py:78,86`) — eigen-decomposition and matrix inverse to build the spectral projectors `P_α = np.outer(V[:, a], Vinv[a, :])`.
- `np.linalg.cond(V)` (`spectral_propagator.py:79`) — the eigenvector condition number, the defectiveness guard.
- `np.linalg.eigvals(ref.Dhat)` (`pipeline_bridge.py:520`) — the Dyson convergence ratio `ρ = max|eig(𝒟̂)|/D₀`.
- `np.polynomial.legendre.leggauss(n_s)` (`dyson_dressing.py:101`) — Gauss–Legendre nodes/weights for the `s`-integral.
- `np.einsum('s,sij,jk,slk->il', s_w, G_late, N, G_early)` (`dyson_dressing.py:110`) — the tensor contraction assembling `∫ds G N Gᵀ` over the quadrature.
- `np.stack`, `np.broadcast_to`, `np.einsum('kp,kx->px', ...)` in the Wick-moment numerics (`pipeline_bridge.py:919-923`).
- `itertools.product`-driven assignment batching (`pipeline_bridge.py:1214-1216`): `np.array(list(itertools.product(range(nf), repeat=n_rows)), dtype=int).T` builds the `(n_rows, n_assign)` assignment grid, then `np.prod(elems[...], axis=0)` collapses it to per-assignment weights.

### SciPy (`scipy.linalg`, `scipy.integrate`)

**What it is.** SciPy is the scientific-computing layer built on top of NumPy. It wraps battle-tested Fortran/C numerical libraries (LAPACK, QUADPACK) behind a Python API. This subsystem uses two corners of it.

**`scipy.linalg`** — dense linear algebra beyond what `np.linalg` offers:
- `from scipy.linalg import expm, solve_continuous_lyapunov` (`spectral_propagator.py:48`).
- `expm(Z)` is the **matrix exponential** `e^Z` (Padé approximation with scaling-and-squaring, robust for non-normal matrices). Used for `Φ_n` via Opitz: `expm(-float(t) * Z)[0, n]` (`spectral_propagator.py:256, 274`), and for `e^{-A|τ|}` in `coupled_two_point` (`spectral_propagator.py:185`).
- `solve_continuous_lyapunov(A, N)` solves the **continuous Lyapunov equation** `A X + X Aᴴ = Q` (here `Q = N`, and `Aᴴ = Aᵀ` since `A` is real). This gives the stationary covariance `Σ` (`spectral_propagator.py:166`). The function uses the Bartels–Stewart algorithm (Schur decomposition + back-substitution), far more stable than inverting the `N²×N²` Kronecker system by hand.

**`scipy.integrate`** — numerical quadrature:
- `from scipy import integrate` (`loop_dyson.py:43`).
- `integrate.quad(f, a, b, limit=...)` is adaptive Gauss–Kronrod quadrature (QUADPACK's `QAGS`/`QAGI`). Used throughout `loop_dyson.py` for the self-energy momentum integrals (`integrate.quad(f, -np.inf, np.inf, limit=120)`, `loop_dyson.py:64,74`) and the Dyson time integrals (`integrate.quad(t1f, 0, np.inf, limit=200)`, `loop_dyson.py:102,105`). `limit` caps the number of adaptive subdivisions.

### SymPy

**What it is.** SymPy is a pure-Python **symbolic mathematics** library (a computer algebra system, CAS). It manipulates mathematical expressions as symbolic trees — symbols, polynomials, derivatives — exactly, not numerically. It is distinct from Sage (below): SymPy is lightweight and pip-installable; Sage is a large CAS distribution. This subsystem uses SymPy for the **momentum-space polynomial algebra** of diagrams (routing momenta, building form factors, taking Wick moments), because that work is polynomial bookkeeping that does not need Sage's heavier algebraic-number machinery.

**How this code uses it.** Always lazily imported as `import sympy as _sp` inside functions (to avoid import-time cost / cycles). Key calls:
- `_sp.expand(v)`, `_sp.expand(e.diff(syms[ii]).diff(syms[jj]))` in `_diagram_is_bubble` (`pipeline_bridge.py:676,680`) — expand a routed edge `k²` and take mixed second partials to detect a `q·ℓ` cross term (bubble vs tadpole).
- `_sp.Integer`, `_sp.I` (the imaginary unit), `_sp.Symbol`, `_sp.symbols('_Xi0:%d' % p)` to mint indexed symbol families, and `_sp.sympify(...)` to coerce expressions, throughout `diagram_form_factor` and `_build_wick_moment`.
- `_sp.Poly(expr, *vars)` — wrap an expression as a dense/sparse polynomial in named variables to read off `.total_degree()`, `.terms()`, `.degree()`, `.coeff_monomial(...)`. E.g. `_sp.Poly(_sp.expand(Rcal), *loop_syms).total_degree()` in `_min_gh_order` (`pipeline_bridge.py:796`), and `_sp.Poly(G, Qv)` then `.terms()` to march over powers of the external-momentum random variable in `_build_wick_moment` (`pipeline_bridge.py:884-886`).
- `_sp.binomial`, `_sp.factorial2`, `comb` (from `math`) for the non-central Gaussian moment `E[Q^m]` and Isserlis pairings (`pipeline_bridge.py:858-861, 802-812`).
- **`_sp.lambdify(args, expr, 'numpy', cse=True)`** (`pipeline_bridge.py:911, 937, 997, 1043, 1979`) — the bridge from symbolic to numeric. `lambdify` *compiles* a SymPy expression into a fast Python function whose body is NumPy operations, so the per-sample form-factor moment can be evaluated on whole arrays at once. `cse=True` runs common-subexpression elimination first (factoring out shared subterms like `1/Bcal^k`), a real speedup for the Wick-moment polynomials.

### Sage (`sage.all`)

**What it is.** SageMath is a large open-source mathematics system that bundles many CASs behind a unified Python interface. The framework uses Sage's **Symbolic Ring** (`SR`) as the canonical symbolic-expression type for the *physics-facing* algebra — the action, the inverse propagator `K_ft`, the mass/diffusion coefficients, parameter substitution. (SymPy, above, is used only for the diagram momentum polynomials.) `SR` expressions support `.subs(...)`, `.expand()`, `.coefficient(...)`, `.simplify_full()`, `.variables()`, and conversion to Python `complex`/`float`.

**How this code uses it.**
- `from sage.all import SR` (`pipeline_bridge.py:50`, `spatial_correlator.py:38`); `from sage.all import SR, I, function` (`spatial_operator_ir.py:59`).
- `SR.var(name)` mints (or fetches the cached) symbol by name — Sage caches symbols globally by name, so `SR.var('Laplacian')` anywhere returns the *same* object (used deliberately: `pipeline_bridge.py:271` `Lap = SR.var('Laplacian')`, `spatial_operator_ir.py:78` `GRADX_SYM = SR.var('GradX', ...)`).
- `_norm_sr` (`pipeline_bridge.py:68`) normalizes a parameter dict to SR-keyed: `SR.var(str(kk))` as keys, used everywhere parameters are substituted.
- Substitution + numeric coercion: `complex(SR(mu_expr).subs(nps_sr))`, `float(...)` (`pipeline_bridge.py:175-177`, etc.). The Laplacian trick: `nps[Lap] = -(qval ** 2)` (`pipeline_bridge.py:272`) injects `Laplacian → -q²` so the shared time-domain pipeline sees a q-parameterized rational propagator.
- `SR(pre).subs(nps)` and `val.free_variables()` in `_prefactor_is_live` (`pipeline_bridge.py:1068`) to test whether a diagram's scalar prefactor is non-zero at the saddle.
- `function('Lap')`, `function('Dt')`, `function('Dx')` (`spatial_operator_ir.py:68-70`) — Sage's `function(name)` creates an **inert symbolic function** (a callable symbol with no defined evaluation), so `Lap(phi)` is a symbolic tree node `Lap(phi)` that survives substitution/printing/tree-walking unchanged. The IR's entire design rests on this: the operators are inert nodes and all semantics live in explicit passes.
- `from sage.symbolic.operators import add_vararg, mul_vararg` (`spatial_operator_ir.py:62`) — Sage's n-ary `+`/`*` "heads," used by `_is_add`/`_is_mul` to recognize sum/product nodes when tree-walking. (Wrapped in `try/except` because the import location is version-dependent.)
- `e.operator()`, `e.operands()`, `e.variables()` — Sage's expression-tree introspection API. `e.operator()` returns the top-level head (a Python operator, a vararg head, or a `function`); `e.operands()` the children; `e.variables()` the free symbols.

### Python standard library

- **`itertools`** (`dyson_dressing.py:38`, plus lazy imports in `pipeline_bridge.py`). `itertools.product(range(nf), repeat=n+1)` enumerates all projector strings `(α_0,…,α_n)` in `hcal_n` (`dyson_dressing.py:60`) and `_hn_labels` (`pipeline_bridge.py:1297`); `itertools.combinations_with_replacement(range(n_rows), order_n)` enumerates the multisets distributing `n` Dyson insertions across rows (`pipeline_bridge.py:1344`); `itertools.permutations` drives the divided-difference and set-partition helpers.
- **`math`** — scalar transcendentals (`math.exp`, `math.pi`, `math.sqrt`, `math.log`, `math.comb`, `math.factorial`) in the per-sample inner loops of `loop_dyson.py` and the Wick-moment construction.
- **`os`** — environment-variable knobs: `os.environ.get('SPATIAL_DYSON_ORDER')`, `'SPATIAL_GRID_NT'`, `'SPATIAL_GRID_NS'`, `'SPATIAL_INTEGRATOR'`, `'SPATIAL_MC_N'`, `'SPATIAL_MEM_BUDGET_GB'`, `'SPATIAL_Q_CUT'`, `'SPATIAL_N_Q'`, `'SPATIAL_FORCE_NUMERICAL_FT'` (all read in `pipeline_bridge.py`).
- **`concurrent.futures.ThreadPoolExecutor`** (`pipeline_bridge.py:1714`) — *thread* (not process) parallelism for the numerical-FT q-loop, smart-gated to engage only at loop order ≥ 2 where the per-task numpy work releases the GIL.
- **`warnings`** (`pipeline_bridge.py:1502`) — to warn that derivative-vertex theories at `max_ell ≥ 2` are correct but expensive.
- **`dataclasses.dataclass`** (`spectral_propagator.py:45`) for `SpectralReference`; the descriptor dataclasses (`CEdge`, `CStackDiagram`, `RoutingResult`) live in their own modules.

### NetworkX, nauty, numba

These do **not** appear directly in this subsystem's five primary files. (Graph/automorphism work — nauty/Sage automorphisms — and any numba acceleration live upstream in enumeration/symmetry and the core integrator, not in the bridge or coupled-field modules.) They are listed here only to record their *absence* from `spatial-coupled`; do not document them in this chapter.

---

## Components

Listed file by file, in roughly the order a reader meets them. Signatures are quoted verbatim where load-bearing.

### `spectral_propagator.py` — the coupled reference propagator

#### `split_reference_diffusion(D_mat, D0=None)` — `spectral_propagator.py:55`
- **Takes:** `D_mat` (the `(N,N)` diffusion matrix), optional scalar override `D0`.
- **Returns:** `(D0, Dhat)` with `Dhat` an `(N,N)` array (zero iff `𝒟 ∝ I`).
- **Does:** If `D0` is None, defaults it to the isotropic part `trace(𝒟)/N` (the mean eigenvalue — a sensible reference that minimizes `‖𝒟̂‖`). Computes `Dhat = D_mat - D0·I`. The override lets a caller minimize `‖𝒟̂‖/D₀` for Dyson convergence.

#### `spectral_projectors(M)` — `spectral_propagator.py:70`
- **Takes:** a diagonalizable matrix `M`.
- **Returns:** `(eigvals (N,) complex, projectors list[(N,N) complex])` with `M = Σ_α m_α P_α`.
- **Does:** `w, V = np.linalg.eig(M)`. Computes `cond = np.linalg.cond(V)`; if not finite or `> _COND_CAP (1e10)`, **raises `ValueError`** ("near-defective: the defective/confluent case is deferred"). Else inverts `V` and builds each projector as the outer product `np.outer(V[:, a], Vinv[a, :])`. This is the standard spectral resolution of a diagonalizable matrix.

#### `class SpectralReference` (dataclass) — `spectral_propagator.py:91`
Cached spectral data for `G₀`. Fields: `M`, `D` (full diffusion), `D0` (scalar reference), `Dhat` (residual), `eigvals` (`m_α`), `projectors` (`[P_α]`). Properties:
- `n_fields` → `M.shape[0]`.
- `is_scalar_diffusion` → `bool(np.allclose(self.Dhat, 0.0))`. **True ⇒ `G₀` is the EXACT full propagator (no Dyson needed).**
- `G0(ksq, t)` → calls `reference_propagator(...)`; the matrix `Σ_α P_α e^{-(m_α+D₀·ksq)t}` (caller applies `Θ(t)`).

#### `build_reference(M, D, D0=None)` — `spectral_propagator.py:116`
- **Takes:** reaction `M`, diffusion `D` (numeric `(N,N)`), optional `D0`.
- **Returns:** a `SpectralReference`.
- **Does:** validates square/same-shape, splits the diffusion, computes the projectors, assembles the dataclass. The single constructor every coupled path uses.

#### `reference_propagator(eigvals, projectors, D0, ksq, t)` — `spectral_propagator.py:130`
- **Returns:** the matrix `G₀(t,k) = Σ_α P_α e^{-(m_α + D₀·ksq)t}` (equivalent to `e^{-Mt}·e^{-D₀·ksq·t}`; at `ksq=0` it is `e^{-Mt}`). A plain loop summing projector-weighted exponentials.

#### `lyapunov_covariance(A, N)` — `spectral_propagator.py:156`
- **Returns:** the stationary covariance `Σ` solving `A Σ + Σ Aᵀ = N`.
- **Does:** one call to `solve_continuous_lyapunov(A, N)`. `A` must be a stable relaxation matrix (eigenvalues with positive real part); `N` symmetric.

#### `coupled_two_point(ref, N, qsq, tau)` — `spectral_propagator.py:169`
- **Takes:** a `SpectralReference`, noise matrix `N`, `qsq = |q|²`, time offset `tau`.
- **Returns:** the free 2-point **matrix** `C(q,τ)` (real `(N,N)`; `C_ij` is the cross-correlation of fields `i,j`).
- **Does:** **Raises** if `not ref.is_scalar_diffusion` (unequal diffusion needs Dyson). Forms `A = M + D₀·qsq·I`, solves the Lyapunov equation for `Σ`, computes `G = expm(-A·|τ|)`, and returns `G @ Σ` (τ≥0) or `Σ @ G.T` (τ<0). This is the tree-level coupled correlator used by the bridge's `compute_coupled_tree_correlator`.

#### `_opitz_bidiagonal(nus)` — `spectral_propagator.py:211`
- **Returns:** the `(n+1)×(n+1)` upper-bidiagonal Opitz matrix `Z` with diagonal `(0, ν_1,…,ν_n)` and ones on the superdiagonal. Built as `np.diag(ones(n), k=1)` then setting the sub-diagonal nodes.

#### `phi_n(t, nus)` — `spectral_propagator.py:221`
- **Takes:** a scalar `t ≥ 0` and a sequence `nus` of (possibly complex) nodes `ν_1…ν_n`.
- **Returns:** a Python `complex` — `Φ_n(t; ν) = (-1)ⁿ · expm(-t·Z)[0, n]`. Empty `nus` ⇒ `Φ_0 = 1`.
- **Does:** builds the Opitz matrix, exponentiates, reads the corner entry. The confluent-safe time-side primitive. **Complex support is required** because a real `M` can have complex spectrum (conjugate-pair eigenvalue differences).

#### `phi_n_batch(ts, nus)` — `spectral_propagator.py:259`
- **Returns:** a complex array of `Φ_n` over an array of `t` (shared nodes), shape-preserving. One `expm` per `t` on the shared tiny `Z`. The vectorized form used by `hcal_n`.

### `dyson_dressing.py` — the Dyson–Duhamel dressing

#### `hcal_n(ts, M, Dhat, n)` — `dyson_dressing.py:47`
- **Takes:** time array `ts`, reaction `M`, residual `Dhat`, order `n`.
- **Returns:** `(n_t, nf, nf)` complex — `𝓗_n(t)` (B27) for the time array.
- **Does:** `n=0` ⇒ `e^{-Mt}`. Otherwise: gets `eig, proj` from `spectral_projectors(M)`, then for every projector string `(α_0,…,α_n)` in `itertools.product(range(nf), repeat=n+1)` builds the outer-product chain `mat = P_{α_0} 𝒟̂ P_{α_1} ⋯ 𝒟̂ P_{α_n}`, skips strings whose matrix is numerically zero (`> 1e-300`), forms the scalar `e^{-m_{α_0}t}·Φ_n(t; ν)` with `ν = m_{α_i}-m_{α_0}` via `phi_n_batch`, and accumulates `scalar ⊗ matrix`. Cost is `nf^{n+1}` strings.

#### `dressed_GR(ts, qsq, M, Dhat, D0, order)` — `dyson_dressing.py:73`
- **Returns:** `(n_t, nf, nf)` complex — the truncated dressed retarded propagator `G_R^{(N)}(t,k)` at one `|k|² = qsq`.
- **Does:** `acc = Σ_{n=0}^{order} (-qsq)ⁿ · 𝓗_n(ts)`, then multiplies by `e^{-D₀·qsq·t}` (B26). `order=0` is the scalar-`D₀` reference `G₀`.

#### `dressed_tree_C(q, tau, M, Dhat, D0, N, order, n_s=48, s_cap_scale=32.0)` — `dyson_dressing.py:87`
- **Returns:** the dressed tree 2-point **matrix** `C^{(N)}(q, τ)`.
- **Does:** validates `min Re eig(M) > 0` (stable theory) — **raises `ValueError`** otherwise. Sets `cap = s_cap_scale / (mu_min + D₀·q²)`. Builds Gauss–Legendre nodes/weights via `leggauss(n_s)`, maps to the `σ`-concentrated substitution `s = cap·v²` with weights `w·cap·v` (the same trick the chamber integrator uses — packs nodes near `s=0`). Evaluates `G_late = dressed_GR(|τ|+s, ...)` and `G_early = dressed_GR(s, ...)`, then `C = np.einsum('s,sij,jk,slk->il', s_w, G_late, N, G_early)`. Returns `C` (τ≥0) or `C.T` (τ<0). Exact in the `order→∞` limit; at `order=0, 𝒟̂=0` reproduces the spectral-Lyapunov `coupled_two_point`.

#### `dressed_tree_C_q_grid(qs, taus, M, Dhat, D0, N, order, n_s=48)` — `dyson_dressing.py:114`
- **Returns:** `(n_tau, n_q, nf, nf)` complex — `dressed_tree_C` over a q×τ grid (the tree driver's FT input). A double loop calling `dressed_tree_C`.

### `loop_dyson.py` — the 1-loop bubble assembly (ORACLE-ONLY)

> **Banner (`loop_dyson.py:1-5`):** "⚠ ORACLE-ONLY — not on the production path. Superseded by `full_integrator.py`; reached only by its own test(s). `compute_cumulants` does NOT use this module." Document it as a cross-check oracle, not a live path.

It implements the 1-loop `φ̃φ²` reaction-diffusion **bubble** self-energy via the MSR Dyson equation, both the equal-time structure-factor correction `δC(q,0)` and the full `τ`-dependent correction.

- `_mk(k, mu, D)` — `loop_dyson.py:46` — the mass `m(k) = μ + D·k²`.
- `sigma_R_time(q, t, mu, D, T, formfactor=None)` — `loop_dyson.py:57` — the retarded bubble `∫dℓ/2π F(ℓ) G_R(ℓ,t) C(q-ℓ,t)` (`t>0`), via `integrate.quad` over `ℓ ∈ (-∞,∞)`. `formfactor` is an optional derivative-vertex factor `F(ℓ)`.
- `sigma_K_time(q, t, mu, D, T, formfactor=None)` — `loop_dyson.py:68` — the Keldysh bubble `∫dℓ/2π F(ℓ) C(ℓ,t) C(q-ℓ,t)` (even in `t`).
- `C_R, C_K = 4.0, 2.0` — `loop_dyson.py:89` — the principled 1-loop normalization (pinned from the framework's own `𝒮(Γ)` values; validated `B=0.99` vs a perturbative simulation).
- `_dyson_terms(q, mu, D, T, formfactor=None)` — `loop_dyson.py:92` — returns the two Dyson terms `(T1, T2)` of `δC(q,0)` (each a closed 1-D integral over the self-energy).
- `bubble_delta_S(q, mu, D, T, g=1.0, formfactor=None)` — `loop_dyson.py:110` — the physical equal-time bubble correction `δC(q,0) = g²·(C_R·T1 + C_K·T2)`. Even in `q`. Excludes the q-independent tadpole mass shift.
- `bubble_delta_phi2(mu, D, T, g=1.0, q_cut=40.0)` — `loop_dyson.py:122` — `δ⟨φ²⟩ = ∫dq/2π δC(q,0)` (momentum integral of the bubble, fast-decaying `~1/q⁴`).
- `_sigma_grids(q, mu, D, T, t_max, n_t, n_l=2600, formfactor=None, formfactor_K=None)` — `loop_dyson.py:133` — tabulates `σ_R(t), σ_K(t)` on a `t`-grid by a single **vectorized** `∫dℓ` trapezoid (≈100× faster than per-`t` quad; matches it to <1e-4). `formfactor_K` lets the Keldysh form factor differ from the retarded one (a derivative vertex has `F_R ≠ F_K`).
- `_sigma_grids_dD(q, mu, D, T, t_max, n_t, spatial_dim, n_l=110, L_cut=None)` — `loop_dyson.py:161` — the `d∈{2,3}` analog via a direct vectorized `∫dᵈℓ` over a Cartesian grid, with a physical UV cutoff `L_cut`.
- `bubble_delta_C_q_tau(q, taus, mu, D, T, g=1.0, t_max=60.0, n_t=4000, formfactor=None, formfactor_K=None, spatial_dim=1, L_cut=None)` — `loop_dyson.py:193` — the full `δC(q,τ)` for all `τ`, via the **time route** (the frequency route converges only as 1/ω because `Σ_R` has a `t=0` step — Gibbs). Each Dyson term collapses to a fast 1-D integral over the tabulated self-energy. Contains a careful adaptive `a`-grid (≥50 points per `1/m`, capped for large-q modes) and a **power-law sliver** treatment of the integrable `a→0⁺` singularity of a derivative-vertex `σ_R(a)~A·a^p` with `p∈(-1,0)`.

### `pipeline_bridge.py` — the dispatcher and assembler

This is the heart of the subsystem. Components grouped by their `## N.` section banners.

#### Helpers (§ helpers)
- `_norm_sr(num_params)` — `pipeline_bridge.py:68` — normalize a parameter dict (str- or SR-keyed) to **SR-keyed**: `{SR.var(str(kk)): vv}`.
- `_field_index(prop, leg_name)` — `pipeline_bridge.py:76` — map an external-leg field name to the diagonal field index in the propagator's physical columns. Matches on exact name or the digit-stripped base.
- `_legs_to_phys_idx(external_fields, phys_idx)` — `pipeline_bridge.py:89` — map external-leg specs to **valid `phys_idx` keys** for diagram enumeration. Crucial Stage-A unblock: an auto-correlation arrives as `[('dphi',1),('dphi',2)]` but a single-population field only has the key `('dphi',1)`, so both legs must land on the same valid key (else `enumerate_unique_diagrams` types 0 diagrams). Tries the natural name and the `d`-prefixed fluctuation name. **Never** falls back silently — raises `SpatialPropagatorError` if a leg names no physical field (a wrong leg would silently compute a different correlator).
- `_bc_from_prop(prop, num_params_sr, model=None)` — `pipeline_bridge.py:131` — resolve `(bc_mode, L)` (boundary condition mode and periodic length) for the diagonal path, reading off the prop but falling back to `model['boundary']` (because `build_propagator` does not always surface `bc_mode`/`bc_params`). Trusts the model whenever it declares periodic (the diagonal prop can carry a stale `bc_mode='infinite'`).

#### § 1. per-mode structure
- `diagonal_modes_from_propagator(prop, ft, num_params, field_index)` — `pipeline_bridge.py:161` — return `[(mu, D, kap)]`, the per-mode heat-kernel structure for one diagonal field. `mu = ac_mass` (relaxation), `D = ac_diffusion` (Laplacian coefficient), `kap` = white-noise spectral weight from `extract_noise_coefficients`. **Guards:** raises on negative diffusion (anti-diffusive/ill-posed), zero diffusion (a time-only field with no heat-kernel mode), and missing noise coefficient. The single-element list shape is what generalizes to a multi-mode dressed propagator.
- `_modes_C_q_tau(modes, qval, taus)` — `pipeline_bridge.py:197` — the reference `C(q,τ) = Σ_α κ_α/(μ_α+D_α q²)·e^{-(μ_α+D_α q²)|τ|}`. The "modes are right" side of certification.

#### § 2. run the shared pipeline at Laplacian = -q²
- `build_pipeline_records(ft, model, prop, external_fields, max_ell=0, k=2, verbose=False, header='[spatial pipeline]')` — `pipeline_bridge.py:208` — enumerate + classify the (q-independent) diagram topology **once**. Lazy-imports the *exact* entry points `pipeline/compute.py` uses: `extract_vertex_types`/`extract_source_types` (core vertices), `build_field_index_map` (type assignment), `classify_coefficient_factors` (symmetry), and `enumerate_unique_diagrams` (the shared enumerator). Returns `{ell: [(typed_diagram, SR scalar_prefactor), …]}` for `compute_correction_td`. This is genuinely the same diagram machinery the temporal path runs.
- `pipeline_C_q_tau(prop, records, external_fields, base_np_sr, qval, taus, k=2)` — `pipeline_bridge.py:259` — run the shared pipeline at `Laplacian = -q²` → `C(q,τ)`. Sets `nps[Lap] = -(qval**2)`, calls `compute_poles_and_residues(prop, nps)` (mutates the prop's pole/residue cache, re-solved per q, exactly as `compute.py` does), slices out `propagator_data`, and calls `compute_correction_td(...)` to get the callable `total_C`, evaluated at each `τ`.
- `certify_modes(modes, prop, records, external_fields, base_np_sr, q_samples, tau_samples, k=2)` — `pipeline_bridge.py:286` — the max relative error between the pipeline's diagram `C(q,τ)` and the per-mode reference over the sample grid. Small ⇒ "the diagrams the pipeline produced ARE the heat-kernel modes the q-FT will transform."

#### § 3. top-level diagonal-tree driver
- `compute_spatial_correlator_via_pipeline(ft, model, prop, num_params, external_fields, tau_grid, spatial_grid, verbose=False, certify=True, q_samples=(0.0,0.7,1.5), tau_samples=(0.5,1.0), certify_tol=1e-8, q_cut=40.0, n_q=2000, enum_verbose=None, stage_headers=False)` — `pipeline_bridge.py:304` — the **drop-in tree-level correlator that routes through the shared pipeline.** Steps:
  1. If the prop has no `G_tx_sym` (diagonal Tier-1 heat-kernel block unavailable, e.g. off-diagonal coupling), **route to `compute_coupled_tree_correlator`**; re-raise with the original reason if that also fails.
  2. Translate external legs to valid phys_idx keys; pick the field index `fi`.
  3. If the two legs name **different** fields (`_cross_legs`), route to the coupled driver (the diagonal path would wrongly reconstruct `⟨φ_iφ_i⟩`). Auto-correlators stay on the fast diagonal path even with off-diagonal noise (`C_ii = N_ii/2μ_i` at tree).
  4. Read the per-mode `(mu,D,kap)`; resolve the boundary.
  5. If `certify`: build the (ell=0) records and compute `certify_max_rel`; **raise `SpatialPropagatorError`** if it exceeds `certify_tol`.
  6. Do the analytic `q→x` FT: at `d=1`, `C(x,τ) = Σ_modes free_two_point(μ,D,κ; x,τ)` (exact erf/heat-kernel, no ringing); at `d≥2`, build `C(q,τ)` on a `q_cut`-truncated grid and apply `radial_inverse_ft` / `periodic_inverse_ft`.
  - **Returns** `(C, info)` with `info['pipeline_certified']`, `info['certify_max_rel']`, `info['modes']`, etc.

- `compute_coupled_tree_correlator(ft, model, prop, num_params, external_fields, tau_grid, spatial_grid, *, q_cut=60.0, n_q=6000, n_modes=600, verbose=False)` — `pipeline_bridge.py:449` — the **coupled-field tree-level `C_ij(x,τ)`** via the spectral-Lyapunov 2-point + a `q→x` FT, for theories whose inverse propagator has off-diagonal coupling but **scalar** diffusion (`𝒟̂=0`). Steps:
  1. Read `prop['K_ft']` (always present even when the diagonal block is rejected); extract `M, 𝒟, V` via `reaction_diffusion_matrices`; numericize. **Require `V=0`** (drift unsupported in scalar-diffusion v1) — raises `NotImplementedError` otherwise.
  2. Build the `SpectralReference`. If `not ref.is_scalar_diffusion` (unequal diffusion): consume the **Dyson policy** (`model['spatial']['dyson']` or the `SPATIAL_DYSON_ORDER` env override); require `mode='fixed'`; set `dyson_order`; check the **convergence ratio** `ρ = max|eig(𝒟̂)|/D₀ < 1` (raise if `≥1`; warn if `>0.6`).
  3. Extract the noise matrix `N`; resolve the external field indices `(i,j)` and the boundary.
  4. Define `_Cq(q,tau)` as either `dressed_tree_C(...)[i,j]` (Dyson order set) or `coupled_two_point(ref, N, q², tau)[i,j]` (scalar diffusion).
  5. The `q→x` FT branches: `d≥2` radial/periodic; `d=1` periodic lattice sum; `d=1` Dyson/forced-numerical brute cosine FT; **else the ANALYTIC spectral IFT** — `C(q,τ) = Σ_{αβ} (P_α N P_βᵀ)/(m_α+m_β+2D₀q²)·e^{-(m_α+D₀q²)τ}`, each `(α,β)` term a single-mode correlator at denominator mass `μ_d=(m_α+m_β)/2`, transformed exactly via `free_two_point(μ_d, D₀, ½; x, τ)`. `τ<0` uses `C_ij(-τ)=C_ji(τ)`.
  - **Returns** `(C, info)` with `info['coupled']=True`, `M`, `D0`, `Dhat`, `N`, `legs=(i,j)`, and `dyson_order` if set.

#### § 4. loop-correlator machinery
- `_diagram_is_bubble(td)` — `pipeline_bridge.py:664` — True iff a 1-loop diagram is a momentum-**dependent** bubble: some edge carries a momentum mixing external `q` and loop `ℓ` (a `q·ℓ` cross term), detected as a nonzero **mixed second partial** of the edge `k²` (via `route_momenta(td).edge_k2()` + sympy `.diff().diff()`). A tadpole has every edge at pure `q²`/`ℓ²`/`0` (no cross term). Topology-agnostic.
- `diagram_form_factor(td, vertex_terms, mode=None, d=1)` — `pipeline_bridge.py:685` — assemble the symbolic momentum-space **form factor `Rcal(q,ℓ)`** for an arbitrary diagram, any `ell`, any `k`, any mix of derivative-vertex types. `Rcal = ∏_{interaction vertices} 𝔉(v)`, each `𝔉(v) = Σ_{matching type} w_t·𝔣_t(v)`. Uses `route_momenta(td)` to get per-edge momenta, splits them into outgoing (response-leg) and incoming (physical-leg) momenta per vertex, and applies `_f_chain` (the operator-chain Fourier kernel: `Lap → -p²` or `-Σ_α p_α²` for `d≥2`; `Dx → i p` or `i p_ax`). `'composite'` mode uses the response-leg momentum; `'perleg'` mode the product over physical legs. Backward-compatible with a bare `op_chain` tuple + `mode=` kwarg (the old single-type call). Returns an expanded sympy expr in the routing symbols.
- `bubble_loop_form_factor = diagram_form_factor` — `pipeline_bridge.py:782` — backward-compatible alias.
- `_min_gh_order(Rcal, loop_syms)` — `pipeline_bridge.py:785` — the **minimal Gauss–Hermite order** that integrates the polynomial `Rcal` exactly over the loop Gaussian. GH(n) is exact for degree ≤ 2n-1; after the Cholesky map a monomial of total loop-degree `D` can put degree `D` on a single `Z`, so `n = ⌈(D+1)/2⌉`. Returns a safe fallback (6) if `Rcal` is not polynomial. This is the cheap exact speedup (e.g. 6 → 2-3, shrinking the GH grid by `(n/6)^L`).
- `_isserlis(idx, Sget)` — `pipeline_bridge.py:802` — the symbolic **Isserlis/Wick moment** `E[∏_a ξ_{idx[a]}]` for a zero-mean Gaussian with covariance `Sget(i,j)`: a recursive sum over all perfect matchings of the index multiset of `∏_pairs Σ`. Odd length ⇒ 0.
- `_build_wick_moment(Rcal, ls, qs)` — `pipeline_bridge.py:815` — the **analytic spatial IFT of a derivative-vertex form factor** by the joint `(ℓ,q)`-Gaussian moment (Case C). The principled one-pass route replacing the polynomial fit. Splits the loop momentum `ℓ = a·q + ξ`, turns the external `q` into a complex Gaussian via the FT source, and writes the per-chamber IFT as `δC(x) = A·K(Bcal,x)·M_F` with `M_F = E_{Q~N(c,s)} E_{ξ~N(0,Σ)}[Rcal(a·Q+ξ, Q)]` — both expectations closed-form Gaussian moments (non-central `E[Q^m]`, Isserlis `E[∏ξ^k]`). Built once per diagram and `lambdify`d. Returns `(moment_x, moment_bessel)` — two numpy callables. `moment_x(a, S, Bcal, xs) → (P, n_x)` with the **key perf factorization**: `EF` is a polynomial in `X` whose coefficients `g_k(a,Σ,Bcal)` are x-independent, so the expensive per-sample `g_k` are evaluated once and contracted with cheap X-powers (≈`n_x` speedup). `moment_bessel` is the **λ-graded** variant for the Bessel-K radial backend (grades `EF` by the radial scaling power `m`). **`k=2` only, `d=1`.**
- `_formfactor_callable(td, vertex_terms, mode=None, d=1)` — `pipeline_bridge.py:958` — the **numpy `Rcal(ell, q)` callable** for the full-diagram integrator, built from `diagram_form_factor`. Generic in `L`, `n_ext`/`k`, the mix of derivative types, and `d`. At `d=1`: lambdifies over loop symbols `lᵢ` and external `qⱼ`; attaches `ff.gh_order_needed` (`_min_gh_order`), `ff.q_poly_deg`, `ff.moment_x`/`ff.moment_bessel` (`_build_wick_moment`, falling back to None on failure), and `ff.moment_x_multi` (`_build_wick_moment_multi` when `len(qs)≥2`). At `d≥2`: parses per-axis symbols `lᵢ_α/qⱼ_α`. The `Rcal == 0` case (a diagram that vanishes by a conservation law) returns `_zero_ff_with_moments(...)` — a form factor whose every evaluation path (q-path GH average, scalar analytic IFT, multivariate IFT, Bessel) sees 0, not a missing-moment error.
- `_prefactor_is_live(pre, num_params, tol=1e-12)` — `pipeline_bridge.py:1058` — True if the diagram's scalar prefactor is nonzero at the saddle. A topological bubble whose prefactor `∝ φ*²` is dead at `φ*=0` and must not trigger the bubble route. Substitutes `num_params` (carries the saddle); if free symbols remain, conservatively returns True.
- `_live_bubbles(records, num_params)` — `pipeline_bridge.py:1074` — the records that are live momentum-dependent bubbles.

#### § 4 (loop drivers)
- `compute_coupled_loop_correlator(ft, model, prop, num_params, external_fields, tau_grid, spatial_grid, C0, tree_info, max_ell=1, verbose=False)` — `pipeline_bridge.py:1090` — **loop corrections for a coupled scalar-diffusion theory via spectral assignments (Dyson 3c).** Steps:
  1. From `tree_info`, read `M, Dhat, D0`. If `Dhat ≠ 0` (unequal diffusion): consume the Dyson policy → `dress_order` (per-edge Dyson dressing of loop insertions, exact at every order, combinatorial cost). **Raise** on periodic boundary or derivative vertices (v1 plain only).
  2. Get `eig, proj`; require `min Re eig > 0`.
  3. Enumerate records via `build_pipeline_records(..., max_ell)`.
  4. For each loop diagram: numericize the prefactor `pv`; skip q-dependent/zero. Map to the C-stack (`diagram_to_cstack`); expand to rows (`spectral_rows`). For each row, read the `(resp,phys)` matrix indices from `CEdge.fpairs`, gather projector elements `elems[r,a] = P_a[pi,ri]`, build the full assignment grid (`itertools.product(range(nf), repeat=n_rows)`), compute per-assignment weights `Wgt = ∏_r elems[r,assign]`, and keep the live ones. Build the mass table `eig[assign]`.
  5. If `dress_order ≥ 1`: build **Dyson dressing patterns** — distribute `n` insertions over rows (multisets via `combinations_with_replacement`), each dressed row getting its `𝓗_{n_r}` label set (`_hn_labels` → `_pf_labels`, the generalized partial-fraction expansion of the Duhamel convolution with the log-derivative Bell expansion for repeated masses). The momentum insertions ride `diagram_kinematic_spectral(insert_rows=...)`.
  6. **External-time orientation** (the `i≠j` fix): for `i==j` the mirror pair dedupes to one record + the `Γ(τ)+Γ(-τ)` retarded completion; for `i≠j` both mirror records survive, each evaluated once at the times matching its own leaf field order (applying the ±τ completion would double-count and symmetrize away the genuine τ-asymmetry of the cross-correlator).
  7. For each `τ`, accumulate `pv·(Wp @ I_spec)` over patterns into `dCx_by_ell[el]`.
  - **Returns** `(np.real(C1), info)` with `info['C_by_order']`, `info['coupled_loop']=True`, `info['n_modes']`, `info['eigvals']`, imaginary-part diagnostics.
- `compute_spatial_correlator_generic(ft, model, prop, num_params, external_fields, tau_grid, spatial_grid, verbose=False, q_cut=30.0, n_q=64, max_ell=1, parallel=True, n_workers=None)` — `pipeline_bridge.py:1434` — **the one genuine single-field loop path.** Steps:
  1. Compute the tree `C0` via `compute_spatial_correlator_via_pipeline(certify=True)`. If `tree_info['coupled']`, route to `compute_coupled_loop_correlator`.
  2. Require a single tree mode (single field) `(mu0, D0, kap0)`.
  3. Build the numeric derivative-vertex form-factor table `vterms` from `ns._operator_ir_vertex_terms` (substitute couplings into the symbolic weights `c_t/Σc`). Warn (not error) for derivative vertices at `max_ell≥2`.
  4. **Drift guard:** for any field with a non-zero saddle drift `V≠0` (a genuine advection `v·∂_xφ`), raise — the drifting propagator is validated only at the heat-kernel oracle level, not yet in the momentum integrator (which uses `m_k=μ+D·k²`). Gradient theories (Burgers/KPZ) where `V→0` at `φ*=0` run fine.
  5. Enumerate records; map every diagram (all `1≤ell≤max_ell`) → `(C-stack descriptor, pv, form-factor callable, ell)`; filter to live (`|pv|>1e-14`).
  6. Build the per-diagram quadrature grids `(nt, ns)` (coarser for higher-`n_C` diagrams; `SPATIAL_GRID_NT/NS` env overrides).
  7. **Integrator backend** switch (`SPATIAL_INTEGRATOR`): `grid` (default deterministic product quadrature), `mc` (importance-sampled Monte-Carlo, bounded memory, biased for derivative vertices), `bessel` (radial-Bessel-K × angular-MC, regularizes the `det M→0` singularity).
  8. **Memory guard:** the per-chamber quadrature is `P = n_t^{n_V}·n_s^{n_C}` samples; at `ell=2` this hits the curse of dimensionality. Computes the worst-diagram `(P, n_x)` complex-array size in GB; if `> SPATIAL_MEM_BUDGET_GB` (default 6) and not MC/Bessel, **raise** with the exact numbers and the four mitigation knobs (lower max_ell, MC backend, coarser grid, raise the budget).
  9. **Analytic vs numerical FT:** `_use_analytic = (all_plain or d==1) and not SPATIAL_FORCE_NUMERICAL_FT`. The analytic path calls `diagram_correlator_x(...)` directly to `δC(x,τ)` (no q-grid, no ringing). The numerical path computes `δC(q,τ)` on a grid (smart-thread-gated at `ell≥2`) then `_ft_to_x` (cosine FT at `d=1`, `radial_inverse_ft` at `d≥2`).
  - **Returns** `(C1 real, info)` with `info['C_by_order']` (cumulative correlator at each loop order from one pass), the `(mu,D,T)` modes, `vertex_mode`, imaginary-part diagnostics, and diagram counts.
- `compute_spatial_kpoint(ft, model, prop, num_params, external_fields, points, max_ell=1, verbose=False, n_t_loop=10, n_s_loop=12)` — `pipeline_bridge.py:1788` — the **k-point spatial cumulant at external events** (general-k single-field path). `points` is `(n_pts, k-1, 2)` of `(x_j, τ_j)` offsets of slots `1..k-1` relative to slot 0. Reads `(mu0, D0)` from the certified 2-point tree of the same field, then sums every enumerated diagram through the k-generic `diagram_correlator_pts` driver (orbit-stabilizer external-Wick architecture, `field_respecting_mappings` + `external_wick_compensation`). Loop descriptors with `n_V≥3` use the reduced `(n_t_loop, n_s_loop)` grid. Points sharing a τ-configuration are batched. Returns `(values (n_pts,), info)` with `info['per_ell']`.
- `_build_wick_moment_multi(Rcal, ls, qs)` — `pipeline_bridge.py:1893` — the **multivariate (k≥3) analytic-IFT form-factor moment**, the `n_ext≥2` generalization of `_build_wick_moment`. The external momenta become a complex Gaussian **vector** `Q~N(c, S_Q)` with `c=(i/2)·Binv·X`, `S_Q=Binv/2`; the loop split is `l_k = Σ_j a_kj Q_j + Xi_k`. The moment is one joint Isserlis pass over the block-diagonal zero-mean vector `(Xi, ζ)` with the means `c_j` as symbols. Returns `moment_x_multi(a, S, Binv, X) → (P, n_x)`.
- `compute_coupled_kpoint(ft, model, prop, num_params, external_fields, points, tree_info, max_ell=1, verbose=False, n_t_loop=10, n_s_loop=12)` — `pipeline_bridge.py:2000` — the **k-point cumulant for a coupled scalar-diffusion theory** (the general-k companion of `compute_coupled_loop_correlator`), built on the same spectral-assignment sum but with the orbit-stabilizer external-Wick mapping sum replacing the k=2 orientation/±τ machinery (they coincide at k=2). **Scope:** scalar diffusion only (`Dhat=0`), plain vertices, infinite boundary. Dedups identical `(times, X)` mapping configurations and accumulates `(mult·pv/comp)·(Wk @ I_spec)`. Returns `(values, info)`.

### `spatial_operator_ir.py` — the operator IR

#### Operator constructors
- `_LAP, _DT, _DX = function('Lap'), function('Dt'), function('Dx')` — `spatial_operator_ir.py:68-70` — the inert Sage symbolic functions (the operator node heads).
- `GRADX_SYM = SR.var('GradX', latex_name=r'\partial_x')` — `spatial_operator_ir.py:78` — the bare *multiplicative* symbol a bilinear `Dx` lowers to (first-derivative analogue of `Laplacian`); the heat-kernel propagator substitutes `GradX → i·k` to read off the drift `V`.
- `Lap(expr)` / `Dt(expr)` / `Dx(expr, i)` — `spatial_operator_ir.py:81/86/91` — the public node constructors.

#### Tree helpers
- `_head(e)` — `:97` — `e.operator()` (the top-level head), guarded.
- `_is_add(e)` / `_is_mul(e)` — `:104/109` — recognize sum/product nodes (Python `add`/`mul` or Sage vararg heads).
- `_op_name(e)` — `:114` — return `'Lap'`/`'Dt'`/`'Dx'` if `e` is one of our operator nodes, else None (reads the head's `.name()`).
- `_prod(factors)` / `_syms(expr, names)` / `_as_names(syms)` — `:129/136/142` — product builder, symbol-dependence test, name-set extractor.

#### The passes
- `apply_linearity(expr, fields, coords=('x','y','z'))` — `spatial_operator_ir.py:147` — **push every operator node through sums and constant coefficients.** `Lap(a·δφ + b·δψ) → a·Lap(δφ) + b·Lap(δψ)` for field/coord-independent `a,b`. A factor depending on a *coordinate* is NOT pulled out (the deferred Leibniz/`f̂(p)` case). `_lin_node` first `.expand()`s the argument so binomial powers become sums, then distributes over `+`, pulls out the constant factor `c` (field- and coord-free), and rebuilds the operator on the remaining fluctuation part. An argument with no fluctuation left (`Lap(φ̄)`, `Lap(φ̄²)`, a pure coordinate coefficient) stays **atomic** (annihilation is `kill_means`' contingent job).
- `expand_about_saddle(expr, replacements, fields=None, coords=('x','y','z'))` — `spatial_operator_ir.py:220` — substitute `field → mean + fluct` per `replacements = {field_sym: (mean_sym, fluct_sym)}` and re-apply linearity, so `Lap(φ̄+δφ) → Lap(φ̄) + Lap(δφ)`. The **mean term is retained** (use `kill_means` separately to drop it). A homogeneous mean `φ̄` is treated as a constant (pulls out of the operator).
- `kill_means(expr, mean_syms, ops=('Lap','Dt','Dx'))` — `spatial_operator_ir.py:245` — **contingent annihilation:** `Op(arg) → 0` whenever `arg` depends only on the `mean_syms` (a spatially homogeneous and/or stationary saddle). Deliberately separate from `apply_linearity` (the algebra is always valid; annihilation is contingent on the saddle being homogeneous). For an inhomogeneous saddle, omit `Lap` so `Lap(φ̄)` survives to cancel the rest of the stationarity condition.
- `to_derived_generators(expr, fluct_syms, prefix='Dg')` — `spatial_operator_ir.py:275` — **lower each atomic `Op(δφ)` / `Op(Op(δφ))` / `Dx(δφ,i)` to a fresh ring generator symbol** so the existing multivariate-Taylor `expand` treats it like an ordinary field (the `u=δφ, v=∇²δφ` trick). Returns `(expr2, genmap)` where `genmap[gen] = (base_expr, op_chain)` and `op_chain` is the bottom-up tuple of applied operators (`(('Lap',),)`, `(('Lap',),('Lap',))` for ∇⁴, `(('Dx',0),)`). Works **bottom-up** (an `_innermost_node` walk + repeated `.subs`): an operator wrapping an already-introduced generator extends that generator's chain (so `Lap(Lap(δφ))` resolves to a single ∇⁴ generator). Operators on a pure non-fluctuation arg get a `__mean__` passthrough to avoid an infinite loop.
- `prepare_action(S, fields, replacements=None, homogeneous=True, coords=('x','y','z'))` — `spatial_operator_ir.py:346` — **compose the passes.** Two authoring conventions: (a) `replacements=None` — the action is already in fluctuation fields → `apply_linearity` + lower; (b) `replacements={field:(mean,fluct)}` — `expand_about_saddle` (substitute + linearity, mean kept), then for a homogeneous saddle `kill_means`, then lower. Returns `(S_gen, genmap)`.
- `fourier_lower(expr, genmap, k, omega=None)` — `spatial_operator_ir.py:376` — substitute every derived generator by its Fourier image `g → form_factor(chain, k, ω)·base`. Used to read the bilinear kernel `K(ω,k)` off the (1,1) sector and to attach per-leg form factors.
- `classify_generators(expr, genmap, fluct_syms)` — `spatial_operator_ir.py:388` — split the derived generators into **bilinear** (appear only in field-degree ≤ 2 terms → fold into `K(ω,k)`) and **derivative-vertex** (appear in some field-degree ≥ 3 term → carry per-leg form factors). A generator's degree contribution is the field-degree of its *base*. (KPZ `φ̃·(∂ₓδφ)²` is 1+1+1=3 → vertex; reaction-diffusion `φ̃·∇²δφ` is degree 2 → bilinear.) Returns `(bilinear, vertex)`.
- `form_factor(chain, k, omega=None)` — `spatial_operator_ir.py:420` — the **Fourier image multiplier** of an operator chain on a leg of momentum `k`, frequency `ω`: `Lap → -|k|²`, `Dt → -iω`, `Dx_i → i k_i`, composed multiplicatively. `'__mean__'` (an un-annihilated mean derivative) → 0.

---

## Data structures

### `prop` — the propagator dict
Built upstream by `build_propagator` / `build_spatial_propagator`. Keys this subsystem reads:
- `'spatial_dim'` (int `d`); `'nf'` (number of response fields, `n_tilde`); `'ring_gen_names'` (list; physical names are `[nf:]`).
- `'ac_mass'` (list/dict by field index → `μ` SR expr — the relaxation rate / `k⁰` term), `'ac_diffusion'` (→ `D`, the `k²`/Laplacian coefficient), `'ac_drift'` (dict field→`V`, the `k¹` drift; 0 = pure heat kernel).
- `'G_tx_sym'` (dict `(i,j)→(A,B)` SR or None — the diagonal Tier-1 heat-kernel block; **`None` ⇒ route to the coupled path**); `'spatial_tier1_error'` (the reason the diagonal block is unavailable).
- `'K_ft'` (the full symbolic inverse-propagator matrix; **always present**, used by the coupled path even when the diagonal block is rejected); `'omega'` (the frequency symbol).
- `'bc_mode'` (`'infinite'`/`'periodic'`), `'bc_params'` (`{'length': ...}`).
- The pole/residue cache keys sliced for `compute_correction_td`: `'K_ker','K_ft','G_ft','adj_ft','D_omega','D_delta','t_var','omega','nf','pole_vals','C_mats'` (re-solved per q by `compute_poles_and_residues`).

### `model` dict
- `model['boundary']` → `{'mode': 'infinite'|'periodic', 'length': str|number}`.
- `model['spatial']['reference_diffusion']` → the user's `D₀` override (`build_reference` default else `trace/N`).
- `model['spatial']['dyson']` → `{'mode': 'fixed'|'off', 'order': int}` (the Dyson truncation policy; `SPATIAL_DYSON_ORDER` env overrides).
- `model['parameters']` → list of `{'name', 'default', ...}` (e.g. for the auto-created PBC length).

### `SpectralReference` (dataclass, `spectral_propagator.py:91`)
`M, D, D0, Dhat, eigvals, projectors`; properties `n_fields`, `is_scalar_diffusion`, `G0(ksq, t)`.

### Per-mode tuple `(mu, D, kap)`
A heat-kernel mode: `mu` (mass, float/complex), `D` (diffusion >0), `kap` (white-noise spectral coefficient). The diagonal driver emits a single-element list `[(mu, D, kap)]`.

### `CEdge` (dataclass, `diagram_descriptor.py:46`)
`a` (loop-momentum coefficient tuple), `b` (external-q coefficient tuple), `kind ∈ {'R','C'}`, `u, v` (endpoint vertex ids; `u==v` ⇒ self-loop tadpole), `external` (bool), `fpairs` (tuple of `(resp_idx, phys_idx)` propagator matrix indices — `((ri,pi),)` for an `R` edge, `((ri_u,pi_u),(ri_v,pi_v))` for a `C` edge; empty for hand-built descriptors). Method `couples_loop()` → True iff `a` is not all zero.

### `CStackDiagram` (dataclass, `diagram_descriptor.py:74`)
`internal_vertices` (interaction-vertex ids, times integrated), `external_legs` (leaf ids in correlation order; leg 0 → external time 0, leg j → τ_j), `edges` (tuple of `CEdge`), `n_loops`. Methods `loop_edges()`, `is_tadpole_like()`.

### `RoutingResult` (dataclass, `momentum_routing.py:50`)
`edge_momenta` (`{(u,v,lbl): sympy expr in q_syms+loop_syms}`), `q_syms`, `loop_syms`, `n_loops`. Methods `edge_k2()` (`{edge: k_e²}`), `edge_coeffs()` (`{edge: (a,b)}` linear routing coefficients).

### Vertex-terms table `ns._operator_ir_vertex_terms`
A list of dicts (built in `theory_compiler.py:799-810`), one per derivative-vertex type:
`{'weight': SR (= c_t/Σc), 'n_phys': int (physical-leg count), 'chain': op_chain tuple, 'mode': 'composite'|'perleg'}`.
The bridge substitutes the couplings to numericize `weight`. Empty/absent ⇒ a plain (non-derivative) theory.

### `records` / `by_ell`
`build_pipeline_records` returns `{ell: [(typed_diagram, SR scalar_prefactor), …]}`. The `typed_diagram` is the shared-pipeline diagram object (has `.prediagram`, `.external_legs`, `.propagator_indices`).

### Form-factor callable `ff` (from `_formfactor_callable`)
A numpy function `ff(ell, q) → array`, decorated with attributes `ff.gh_order_needed` (int), `ff.q_poly_deg` (int), `ff.moment_x` / `ff.moment_bessel` (the analytic-IFT callables or None), `ff.moment_x_multi` (k≥3 multivariate, or None).

### `info` dict (output)
Returned alongside `C`. Common keys: `'modes'`, `'field_index'`/`'legs'`, `'bc_mode'`, `'L'`, `'spatial_dim'`, `'pipeline_certified'`, `'certify_max_rel'`, `'coupled'`, `'M'`, `'D0'`, `'Dhat'`, `'N'`, `'dyson_order'`, `'one_loop'`, `'max_ell'`, `'C_by_order'` (cumulative `C(x,τ)` per loop order), `'n_diagrams'`/`'n_live_diagrams'`, `'max_abs_imag'`/`'imag_frac'` (the imaginary-part diagnostic), `'per_ell'` (k-point), `'eigvals'`/`'n_modes'` (coupled).

---

## Data flow

**Inputs (every driver):** `ft` (expanded `FieldTheory`), `model` (the model dict), `prop` (the propagator dict), `num_params` (str- or SR-keyed parameter→value, includes the saddle e.g. `phistar1` and any PBC length), `external_fields` (the leg specs — for a 2-point auto-correlation, `[('dphi',1),('dphi',2)]`), `tau_grid`, `spatial_grid` (1-D arrays).

**Dispatch logic (the key branch points):**
- `prop['G_tx_sym'] is None` ⇒ off-diagonal coupling ⇒ coupled path.
- `_field_index(leg0) != _field_index(leg1)` ⇒ a cross-correlator ⇒ coupled path.
- `ref.is_scalar_diffusion` (i.e. `Dhat==0`) ⇒ exact `coupled_two_point`; else consume the Dyson policy ⇒ `dressed_tree_C`.
- `max_ell ≥ 1` ⇒ a loop driver; `tree_info['coupled']` ⇒ the coupled loop driver.
- `len(external_fields) ≥ 3` ⇒ a k-point driver.

**The Laplacian substitution (the certification engine).** `pipeline_C_q_tau` sets `nps[Lap] = -(qval**2)` (`pipeline_bridge.py:271-272`), so the *time-only* shared pipeline sees a rational propagator at effective mass `m(q)=μ+D q²`. The diagram answer `C(q,τ)` is then compared against `_modes_C_q_tau` (the heat-kernel modes). Example: for a single-field reaction-diffusion theory, `certify_max_rel` is ~1e-15 (exact at tree level) and `certify` passes.

**Outputs:** `(C, info)` where `C` is `[len(tau_grid), len(spatial_grid)]` (real for the physical correlator; complex form factors leave a residual imaginary part recorded in `info['max_abs_imag']` then projected out by the even cos/radial transform). The k-point drivers return `(values (n_pts,), info)`.

**Concrete coupled-tree example (`compute_coupled_tree_correlator`):**
```
prop['K_ft'] (symbolic matrix)
  → reaction_diffusion_matrices → (M_sr, D_sr, V_sr)
  → numericize → M, Dm, Vm  (require Vm≈0)
  → build_reference(M, Dm, D0) → SpectralReference
  → extract_noise_matrix(ft) → N
  → _Cq(q,τ) = coupled_two_point(ref, N, q², τ)[i,j]   (or dressed_tree_C if Dyson)
  → analytic spectral IFT  Σ_{αβ} (P_α N P_βᵀ)/(m_α+m_β+2D₀q²) · free_two_point(μ_d, D₀, ½; x, τ)
  → C[len(tau), len(x)]
```

---

## Gotchas & caveats

- **macOS fork crash (the big one).** Fork-based parallelism was *removed* from the spatial path. Forking a Jupyter kernel on macOS after matplotlib/Cocoa/BLAS init crashes the kernel and the OS — even with one worker. The only safe speedup is *thread*-based (numpy releases the GIL) or q-grid numpy batching. The numerical-FT path uses a **smart-gated `ThreadPoolExecutor`** (`pipeline_bridge.py:1713`): threads engage only when a loop-order≥2 diagram is present (≈2.5× at L=2; *slower* at L=1 tiny matrices, so L=1 stays serial), and accumulation happens on the main thread for bit-identical results. See the module note at `pipeline_bridge.py:1083-1087`.

- **The memory guard / curse of dimensionality.** A chamber's quadrature is `P = n_t^{n_V}·n_s^{n_C}` samples. A KPZ 2-loop diagram (`n_V=4, n_C=3`) gives `P ≈ 1.8e8` at `(16,14)`, and the `(P, n_x)` heat-kernel array alone is tens of GB → an OOM that crashes the machine. The guard (`pipeline_bridge.py:1633-1665`) refuses up front with the numbers and four knobs (lower `max_ell`, MC backend, coarser `SPATIAL_GRID_NT/NS`, raise `SPATIAL_MEM_BUDGET_GB`). It is bypassed for the MC/Bessel backends (bounded memory).

- **Certification is a hard gate, not a warning.** `compute_spatial_correlator_via_pipeline` *raises* `SpatialPropagatorError` if `certify_max_rel > certify_tol` (1e-8). The message says the `(mu,D,kap)` extraction or diagram routing is wrong for the theory.

- **Defective reaction matrix.** `spectral_projectors` raises `ValueError` if the eigenvector condition number exceeds `_COND_CAP=1e10` (repeated eigenvalues with Jordan blocks). The confluent/resolvent form is deferred. Note `Φ_n` is confluent-safe *in the node differences* (Opitz), but `M`'s own diagonalizability is still required.

- **Dyson divergence guard.** The series converges only if `ρ = ‖𝒟̂‖/D₀ < 1`. `compute_coupled_tree_correlator` raises if `ρ ≥ 1` ("the large-|k| insertion outgrows the reference") and warns if `ρ > 0.6` (slow convergence). The reference `D₀` can be retuned via `.reference_diffusion()` to shrink `ρ`. The series is exact at `q=0` (the insertion vanishes) regardless.

- **Drift `V≠0` is refused (single-field generic path).** A genuine advection `v·∂_xφ` (constant drift surviving at the saddle) is validated only at the heat-kernel oracle level, not in the momentum integrator (`m_k=μ+D·k²`). Only `φ*=0` gradient theories (Burgers/KPZ, where `V→0`) run end-to-end (`pipeline_bridge.py:1540-1554`). The coupled-tree path likewise requires `V=0`.

- **Never default a leg index silently.** Both `_legs_to_phys_idx` and `_leg_idx` raise rather than fall back — a wrong leg would quietly compute a *different* correlator (`C_aa` instead of `C_ab`). Emphasized twice in comments (`pipeline_bridge.py:122-124, 542-543`).

- **`_diagram_is_bubble` is topology-agnostic but dead-bubble-aware.** A topological bubble whose prefactor `∝ φ*²` is dead at `φ*=0` and must *not* trigger the bubble route — `_prefactor_is_live` filters those out. Live bubbles only.

- **The `Rcal == 0` form factor must still evaluate to 0 on every path.** A diagram that vanishes by a conservation law (e.g. the `∇²(φ²)`/`(∂φ)²` tadpole with a forced-zero leg momentum) returns `_zero_ff_with_moments`, which attaches zero-returning `moment_x`/`moment_x_multi`/`moment_bessel` callables — otherwise the analytic IFT would hit a missing-moment error instead of contributing 0 (`pipeline_bridge.py:972-990`).

- **Cross-correlator τ-asymmetry (the `i≠j` fix).** For `i==j` the mirror pair dedupes to one record + the `Γ(τ)+Γ(-τ)` completion. For `i≠j` *both* mirror records survive and each must be evaluated once at the times matching its own leaf order — applying the ±τ completion there would double-count and symmetrize away the genuine asymmetry of `C_ij` (`pipeline_bridge.py:1368-1400`). The coupled-loop and coupled-kpoint drivers handle this differently (orbit-stabilizer mapping sum vs explicit orientation), and the comment notes they coincide at k=2.

- **`loop_dyson.py` is ORACLE-ONLY.** Superseded by `full_integrator.py`; `compute_cumulants` does not use it. Its `C_R=4, C_K=2` normalization and the `B=0.99` validation are historical context, not the live computation.

- **The `2^{-n_C}` factor convention.** The single-field generic path multiplies by the universal `2^{-n_C}` (one-segment C convention). The coupled spectral-assignment paths do **not** — the two-glued-segment C representation natively produces the Lyapunov `1/(m_α+m_β+2Dk²)` denominator, so the conversion does not apply (`spectral_rows` docstring, `full_integrator.py:730-737`; reiterated in the coupled drivers).

- **Wick-moment scope.** `_build_wick_moment` is `k=2, d=1` only (single external `q` symbol). `_build_wick_moment_multi` covers `k≥3` (`n_ext≥2`). `d≥2` derivative-vertex transverse handling is Phase 3 → still numerical FT (`pipeline_bridge.py:1668-1671`).

- **Env-var escape hatches.** `SPATIAL_FORCE_NUMERICAL_FT=1` keeps the numerical-FT cross-check reachable (exact only in `q_cut→∞, n_q→∞`); `SPATIAL_Q_CUT`/`SPATIAL_N_Q` tune it. `SPATIAL_DYSON_ORDER` overrides the model's Dyson policy. These are for cross-checks/debugging, not normal use.

- **The IR's two-pass split is deliberate.** `apply_linearity` (always valid algebra) and `kill_means` (contingent on a homogeneous/stationary saddle) are separate passes precisely so an *inhomogeneous* saddle (a front, a pattern) keeps its `Lap(φ̄)` term, which cancels the rest of the stationarity condition rather than being silently dropped (`spatial_operator_ir.py:36-41`).

- **Deferred IR features.** The Leibniz/product rule for position-dependent coefficients and the `f̂(p)` momentum injection, and operators nested *inside products* beyond a single base (`Lap(Lap(δφ)·ψ)`), are documented-not-implemented (`spatial_operator_ir.py:51-53`). A position-dependent coefficient is left atomic by `apply_linearity`.

---

## Glossary

- **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the field-theoretic (response-field doubling) formulation of a stochastic PDE, turning noise averages into a path integral with a "physical" field `φ` and a "response" field `φ̃`.
- **Reaction (mass) matrix `M`** — the `k⁰` (momentum-independent) part of the linearized inverse propagator; encodes relaxation rates and linear field couplings. Diagonal `M` = decoupled fields.
- **Diffusion matrix `𝒟`** — the `k²` part; encodes spatial spreading. `𝒟 ∝ I` = scalar/equal diffusion.
- **Reference diffusion `D₀`** — the scalar part split off `𝒟 = D₀·I + 𝒟̂`; chosen (default `trace/N`) to make `𝒟̂` small for Dyson convergence.
- **Residual diffusion `𝒟̂`** — `𝒟 − D₀·I`; the Dyson perturbation. Zero ⇒ no series needed.
- **Spectral projector `P_α`** — the rank-1 (or higher) matrix projecting onto the α-th eigenspace of `M`; `M = Σ m_α P_α`. Built from eigenvectors.
- **Retarded propagator `G_R`** — the causal Green's function `Θ(t)·e^{-(M+𝒟k²)t}`.
- **Dyson–Duhamel series** — the order-by-order expansion of `G_R` in powers of the residual insertion `𝒟̂|k|²`; each order is an n-fold Duhamel (time-ordered) convolution.
- **Duhamel convolution** — the time-ordered nested integral expressing a perturbed evolution as a series of insertions sandwiched between unperturbed evolutions.
- **Divided difference `f[x₀,…,x_n]`** — a classical finite-difference object; here the n-fold simplex time integral equals `(-1)ⁿ` times the n-th divided difference of `f(z)=e^{-tz}`.
- **Hermite–Genocchi formula** — expresses a divided difference of `f` as a simplex integral of `f⁽ⁿ⁾`; the bridge between the Duhamel integral and the divided difference.
- **Opitz theorem** — a divided difference equals the top-right entry of `f` applied to the upper-bidiagonal "Opitz matrix" (nodes on the diagonal, ones on the superdiagonal). Confluent-safe (no 0/0 at repeated nodes).
- **`Φ_n(t; ν)`** — the confluent-safe nested-simplex time primitive of the Dyson dressing; `Φ_0=1`, `Φ_1=(1-e^{-tν})/ν`.
- **`𝓗_n(t)`** — the matrix factor of the n-th Dyson order (B27): a sum over projector strings times `e^{-m_{α_0}t}·Φ_n`.
- **Lyapunov equation** — `A Σ + Σ Aᵀ = N`; its solution `Σ` is the stationary covariance of a linear SDE driven by noise covariance `N`.
- **Fluctuation–regression (Onsager regression) theorem** — the statement that the relaxation of an equilibrium fluctuation follows the same law as a macroscopic perturbation: `C(τ) = e^{-Aτ}Σ`.
- **Heat kernel** — the Green's function of the diffusion operator; in `d` dimensions `(4πBt)^{-d/2} e^{-|x|²/4Bt}`. The `q→x` Fourier transform of a mode `κ/(μ+Dq²)e^{-(μ+Dq²)τ}`.
- **Form factor `Rcal(q,ℓ)`** — the polynomial momentum factor a derivative vertex puts on a diagram; `∏_vertices 𝔉(v)`.
- **Operator IR** — the small intermediate representation hosting `∇²`(`Lap`), `∂_t`(`Dt`), `∂_{x_i}`(`Dx`) as inert symbolic nodes with explicit algebra passes.
- **Derived generator** — a fresh ring-generator symbol standing for `Op(δφ)` (the `u=δφ, v=∇²δφ` trick), so multivariate Taylor treats it like an ordinary field.
- **Composite vs perleg vertex** — `'composite'`: the operator acts on the `φⁿ` composite (response-leg momentum; `∇²(φ²)`, `∂_x(φ²)`). `'perleg'`: the operator acts on each physical leg (`(∂_xφ)²`, `∏ i·p_leg`).
- **Symanzik polynomials (`U`, `F`)** — the graph polynomials parameterizing a Feynman loop integral after introducing Schwinger/Feynman parameters; here used for the `∫dᵈℓ` momentum reduction.
- **Causal chamber** — a region of the internal-time integration domain with a fixed time-ordering; the loop integral is a sum of chamber integrals.
- **Gauss–Legendre / Gauss–Hermite quadrature** — fixed-node numerical integration rules; Legendre on a finite interval (the `s`-integral, the chamber times), Hermite against a Gaussian weight (the loop-momentum form-factor average). GH(n) is exact for polynomials of degree ≤ 2n-1.
- **Isserlis / Wick theorem** — the moment of a product of jointly-Gaussian variables equals the sum over perfect matchings of products of covariances; the closed-form loop-momentum average.
- **`C-stack` / `CEdge` / `CStackDiagram`** — the integrator's view of a typed diagram: edges tagged `R` (retarded) or `C` (correlation), with loop/external momentum routing coefficients and (coupled-field) propagator matrix indices `fpairs`.
- **`fpairs`** — per-edge `(response_index, physical_index)` matrix-index pairs threaded into the spectral-assignment sum.
- **`𝒮(Γ)` (symmetry factor)** — the diagram's combinatorial weight; `pv` in the code is `𝒮(Γ)·prefactor`.
- **`SR` (Sage Symbolic Ring)** — Sage's symbolic-expression type; the framework's canonical physics-facing algebra.
- **`lambdify`** — SymPy's compile-a-symbolic-expression-into-a-fast-numpy-function utility.
- **`expm`** — the matrix exponential (`scipy.linalg.expm`).
- **`solve_continuous_lyapunov`** — SciPy's Bartels–Stewart solver for `AX+XAᴴ=Q`.
- **GIL** — Python's Global Interpreter Lock; numpy heavy ops release it, which is why thread (not fork) parallelism is safe here.

---

## Proposed manual subsections

1. **Why coupled fields need a different propagator** — from `e^{-Mt}` to the spectral resolution; the non-commuting `M`, `𝒟` problem.
2. **The reference-diffusion split** — `𝒟 = D₀·I + 𝒟̂`, scalar-diffusion exactness, and the diagonal-reduction sanity check.
3. **Spectral projectors and `G₀`** — building `P_α`, the defectiveness guard, the `SpectralReference` cache.
4. **The tree-level coupled 2-point** — the Lyapunov equation, fluctuation–regression, `coupled_two_point`.
5. **The Dyson–Duhamel series** — the insertion expansion, the Duhamel convolution, `G_n`/`𝓗_n` (B26/B27).
6. **`Φ_n`: divided differences, Hermite–Genocchi, and the Opitz theorem** — the one new primitive, worked `n=1` example, confluent safety.
7. **Dressed tree and dressed loops** — `dressed_GR`, `dressed_tree_C`, the σ-concentrated quadrature, the convergence ratio `ρ`.
8. **Spectral assignments at loop order** — turning one coupled diagram into a weighted sum of scalar diagrams; `fpairs`, the assignment grid, the `2^{-n_C}` convention.
9. **The operator IR** — inert nodes, the algebra (linearity), saddle expansion, contingent mean annihilation, lowering to derived generators, classification.
10. **Form factors and the Fourier rules** — `Lap→-k²`, `Dx→ik`, composite vs perleg, the diagram-level `Rcal`.
11. **The analytic spatial IFT** — the joint `(ℓ,q)`-Gaussian moment, Isserlis, the Wick-moment construction and its `lambdify` perf factorization; the Bessel λ-grading.
12. **The bridge architecture** — the Laplacian-substitution trick, "route through the shared pipeline then certify," the dispatch tree.
13. **The single-field generic loop path** — the one genuine integrator, backends (grid/mc/bessel), the analytic-vs-numerical FT gate, the drift guard.
14. **General-k cumulants** — `compute_spatial_kpoint`/`compute_coupled_kpoint`, external events, the orbit-stabilizer Wick mapping sum.
15. **Operational guardrails** — the memory budget, the macOS fork prohibition, the certification gate, the leg-index safety, the env knobs.
16. **The oracle modules** — `loop_dyson.py` as an independent cross-check (and why it is off the production path).

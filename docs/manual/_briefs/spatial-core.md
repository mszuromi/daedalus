# Spatial Integrator: Symanzik polynomials & causal chambers

**Slug:** `spatial-core`
**Subsystem:** the one genuine spatial Feynman-diagram integrator — every enumerated diagram (tree, bubble, tadpole, sunset; any number of external legs `k`, any loop order `ℓ`, any spatial dimension `d`) is reduced to the *same* integral and evaluated.

Primary files (all under `msrjd/integration/spatial/`):

| file | role |
|---|---|
| `diagram_descriptor.py` | map a typed diagram → the flat "C-stack" edge list (`diagram_to_cstack`) |
| `momentum_routing.py` | solve momentum conservation → per-edge routing coefficients `(a_e, b_e)` |
| `spatial_reduce.py` | the textbook Symanzik momentum reduction (single-sample, reference impl.) |
| `causal_chambers.py` | the causal time-ordering chambers of the internal-vertex time integral |
| `full_integrator.py` | the production integrator: batched Symanzik + chamber quadrature + analytic IFT |
| `generic_evaluator.py` | **ORACLE ONLY** — an older Dyson-convolution evaluator kept as a cross-check |

---

## 1. Overview

### What problem this subsystem solves

We compute connected correlation functions (cumulants) of a stochastic field theory written in the **Martin–Siggia–Rose–Janssen–De Dominicis (MSR-JD)** formalism. After the upstream machinery has (a) expanded the action into vertices, (b) enumerated all topologically distinct Feynman diagrams up to a chosen loop order, and (c) "typed" each diagram (decided which lines are response/retarded `R` lines and which are noise/correlation `C` lines, and assigned a symbolic prefactor that carries couplings, noise amplitudes, and the symmetry factor) — what remains is a hard, multi-dimensional integral *per diagram*:

> integrate over the internal interaction-vertex times `{t_v}`, over the loop momenta `{ℓ_i}`, and over the spatial / noise-source degrees of freedom, with the external times `{τ_j}` and external momenta `{q_j}` held fixed.

This subsystem performs **that integral**, and it does so with *no per-topology special-casing*. A tree, a one-loop bubble, a one-loop tadpole, a two-loop sunset — all become one Gaussian momentum integral (collapsed analytically into the **Symanzik polynomials** `U` and `F`) times a residual **causal-time** integral (done by quadrature over **ordering chambers**), times an analytic inverse Fourier transform back to position space. The phrase the codebase repeats is "ONE genuine integral evaluates *every* enumerated diagram" (`full_integrator.py:4`).

### Where it sits in the end-to-end pipeline

```
   theory (.theory.py)
        │  action → vertices
        ▼
   diagram enumeration  (msrjd/diagrams/…)   → prediagrams
        │  field-type assignment
        ▼
   TypedDiagram          (msrjd/diagrams/type_assignment.py)
        │  .prediagram, .vertex_assignments, .propagator_indices, scalar_prefactor
        ▼
 ┌─────────────────────────  THIS SUBSYSTEM  ──────────────────────────┐
 │  diagram_to_cstack(td)              → CStackDiagram  (descriptor)    │
 │  route_momenta(td) → edge_coeffs()  → per-edge (a_e, b_e)           │
 │  symanzik_*  / _momentum_factor_batch → U, F (the L-loop ∫dℓ)        │
 │  causal_chambers(...)               → orderings of {t_v}            │
 │  diagram_kinematic(...)             → ∫dt ∫dσ (chamber quadrature)   │
 │  _heat_kernel_x_general / IFT       → δC(x,τ) directly (no q-grid)   │
 └─────────────────────────────────────────────────────────────────────┘
        │  Σ over diagrams, × 2^{−n_C}·𝒮(Γ)·prefactor
        ▼
   C(q,τ) or δC(x,τ)   →   pipeline_bridge.py   →   compute_cumulants(...)
```

**Feeds it (input):** a `TypedDiagram` `td` (from `msrjd/diagrams/type_assignment.py`) plus a numeric **prefactor value** `𝒮(Γ)·prefactor` evaluated at the working point (the saddle / numeric couplings). The orchestrator that prepares these is `msrjd/integration/spatial/pipeline_bridge.py` (functions `build_pipeline_records`, `compute_spatial_correlator_*`).

**Consumes its output (downstream):** `pipeline_bridge.py` sums the per-diagram contributions into `δC(q,τ)` / `δC(x,τ)` and hands the result to the user-facing `compute_cumulants` API. The brief's functions `correlator_2pt`, `correlator_2pt_x`, and `diagram_correlator_pts` are the summation entry points.

### The one mental model

A heat-kernel propagator in the `(k, t)` (momentum, time) representation is
`G(k, t) = θ(t) e^{−(μ + D k²) t}`. The crucial fact: **the edge times are already Schwinger parameters**. So the loop-momentum integral `∫ dᵈℓ` is a *pure Gaussian* — it collapses exactly to the Symanzik polynomials at any loop order `L` and any dimension `d`. After that collapse the only thing left is a smooth integral over the internal vertex times and the noise-source Schwinger parameters, which has no pole / close-pair pathology (the pathology lived in the momentum integral, which we did analytically). That is the entire design.

---

## 2. The math

This section builds the relevant theory from the ground up. A reader who knows MSR-JD field theory but not the specific representation should be able to follow.

### 2.1 The heat-kernel propagators

Start from a linearized (Gaussian / "tree") theory with one field. In momentum–time space the **retarded propagator** (the response function, the `R` line) is

```
   G_R(k, t) = θ(t) · e^{−m(k) t},        m(k) = μ + D k²,
```

where `μ` is a mass/relaxation rate, `D` a diffusion constant, `k²=|k|²` the squared momentum, and `θ` the Heaviside step (causality: a response only to the past). The **correlation function** (the `C` line, `⟨φφ⟩`) of the *tree* theory is

```
   C₀(k, t) = (T/m(k)) · e^{−m(k)|t|}.
```

`T` is the noise strength (a `2T φ̃φ̃` noise vertex in the action). Real-space correlators are inverse Fourier transforms of these over `k`.

The key identity the whole subsystem rests on: a correlation line can be written as **two retarded segments glued at a noise source**, with the source time integrated out:

```
   C₀(Δt) = (T/m) e^{−m|Δt|} = T · ∫₀^∞ dσ e^{−m(|Δt| + 2σ)} · (something) ,
```

i.e. the `C` line is an `R`–noise–`R` object, and integrating the noise-source time produces a single **Schwinger parameter** `σ ∈ [0, ∞)` and the weight `e^{−m(|Δt|+σ)}` (the convention factor of 2 is bookkept by `2^{−n_C}`, see §2.6). This is why the enumerator can work entirely with all-`G_R` propagators plus explicit noise sources, and why `diagram_to_cstack` re-contracts a 2-point noise source back into one `C` edge.

### 2.2 Edge weights `w_e` and the loop-momentum integral

For a diagram with internal vertices at times `{t_v}` (to be integrated), external leaves at fixed times `{τ_j}`, and edges `e`, each edge `e` carries a **Schwinger weight** `w_e`:

```
   R edge (head time t_head, tail time t_tail):  w_e = t_head − t_tail   (≥ 0 by causality)
   C edge between times t_a, t_b:                w_e = |t_a − t_b| + σ_e  (σ_e ∈ [0,∞))
```

(`full_integrator.py:16-19`). Each edge also carries a **routed momentum** `k_e` (next subsection). The diagram's loop-momentum integral, in the heat-kernel representation, is the **pure Gaussian**

```
   I_mom = ∫ ∏_{i=1}^{L} dᵈℓ_i / (2π)ᵈ  exp[ −D Σ_e w_e k_e² ],
```

with `L` = number of loops. This is the central object (`spatial_reduce.py:12`).

### 2.3 Momentum routing: `k_e = a_e·ℓ + b_e·q`

A spatial interaction vertex is *local in space*, so `∫dx_v` of the product of fields at the vertex enforces **momentum conservation** `Σ_e k_e = 0` at every internal vertex (the spatial analogue of the time-domain vertex that carries one integrated time `t_v`). Solving that linear conservation system writes each edge momentum as a linear form

```
   k_e = Σ_{i=1}^{L} a_{ei} ℓ_i + Σ_{j=1}^{n_ext} b_{ej} q_j,
```

in the **loop momenta** `ℓ_i` (the `L` free parameters of the conservation system) and the **external momenta** `q_j` (one per external leg, with overall conservation `Σ_j q_j = 0` imposed by eliminating the last leaf so `n_ext = k − 1`). The coefficients `(a_e, b_e)` are the per-edge **routing coefficients**. For a translation-invariant routing they are integers, `±1` or `0`. This is exactly what `momentum_routing.route_momenta` computes and `RoutingResult.edge_coeffs()` returns.

Two illustrative cases (from `momentum_routing.py:29-34`):

* **Tree (`L = 0`).** Every edge carries `±q` only, so `k_e² = q²` uniformly. The single-mode tree therefore needs only a `Laplacian → −q²` substitution.
* **One-loop bubble.** Two internal lines carry `ℓ` and `q − ℓ` respectively — genuinely different momenta — so the substitution becomes per-edge.

### 2.4 Symanzik polynomials from scratch

Collect the loop momenta into the quadratic form. Define three matrices by summing the routing coefficients weighted by the Schwinger weights (`spatial_reduce.py:18-21`):

```
   Lam_{i i'} = Σ_e w_e a_{ei} a_{ei'}      (L × L)            — first Symanzik kernel
   N_{i j}    = Σ_e w_e a_{ei} b_{ej}      (L × n_ext)         — loop/external cross block
   Q_{j j'}   = Σ_e w_e b_{ej} b_{ej'}     (n_ext × n_ext)     — pure external block
```

Then `Σ_e w_e k_e² = ℓᵀ Lam ℓ + 2 qᵀ Nᵀ ℓ + qᵀ Q q`. Completing the square in `ℓ` and doing the Gaussian integral `∫ dᵈℓ/(2π)ᵈ exp(−D ℓᵀ Lam ℓ − …)` gives the closed form (`spatial_reduce.py:24-26`):

```
   I_mom = (4πD)^{−Ld/2} · U^{−d/2} · exp[ −D · qᵀ Q_eff q ],
```

with

```
   U      = det Lam                         — the FIRST Symanzik polynomial (a.k.a. the "U polynomial")
   Q_eff  = Q − Nᵀ Lam⁻¹ N                  — the reduced external quadratic form
   F      ≡ U · Q_eff (scalar case)          — the SECOND Symanzik polynomial / F = U·Q_eff
```

What these polynomials *are*, classically (worth stating because the reader may know the graph-theory definition):

* **`U` (first Symanzik / first graph polynomial).** A homogeneous degree-`L` polynomial in the edge weights. Graph-theoretically, `U = Σ_{spanning trees T} ∏_{e ∉ T} w_e` — sum over spanning trees of the product of weights of the edges *not* in the tree. In this code it is computed *numerically* as `det Lam`, which equals that combinatorial sum (the matrix-tree theorem). Validated examples (`spatial_reduce.py:36-38`): the 1-loop bubble has `U = w₁ + w₂`; the 2-loop sunset has `U = w₁w₂ + w₂w₃ + w₃w₁`.
* **`F` / `Q_eff` (second Symanzik / second graph polynomial).** Carries the external-momentum dependence. `Q_eff = Q − Nᵀ Lam⁻¹ N` is the **Schur complement** of `Lam` in the full quadratic form — exactly the residual external quadratic form after integrating the loops. For the bubble, `Q_eff = w₁w₂/(w₁+w₂)`; for the sunset, `Q_eff = w₁w₂w₃ / U` (`spatial_reduce.py:37`). For `n_ext = 1` (the 2-point function, `k = 2`) `Q_eff` is a scalar and `F = U · Q_eff`.

The exponent `−Ld/2` and `−d/2` are the *only* places the spatial dimension `d` enters the momentum reduction — the reduction is otherwise exact and `d`-agnostic.

### 2.5 The full diagram integral

Putting the pieces together, one diagram's contribution to the connected `k`-point cumulant is (`full_integrator.py:15-20`):

```
   Γ(q, {τ}) = 2^{−n_C} · 𝒮(Γ) · ∫ ∏_v dt_v ∏_{C edges} dσ_e  𝟙(θ's) ·
                              e^{−μ Σ_e w_e} · MomFactor(w, q),

   MomFactor = (4πD)^{−Ld/2} · U(w)^{−d/2} · e^{−D qᵀ Q_eff(w) q}.
```

The factor `e^{−μ Σ_e w_e}` is the **mass damping** (every segment of length `w` decays as `e^{−μ w}`; the `D k²` part of the mass already lives inside `MomFactor` via the Symanzik exponent). `𝟙(θ's)` are the causal step functions of the retarded edges. The residual integral is over the internal vertex times `t_v` and the correlation Schwinger parameters `σ_e` only — and it is **smooth**, because the loop integral (the source of the close-pair pole pathology) was done analytically.

### 2.6 Normalization — derived, not fitted

The integrator's normalization is *derived* (`full_integrator.py:27-31`). The enumeration prefactor uses the `2T` noise-vertex convention; a kinematic `C` edge here is the unit-amplitude Schwinger factor `∫dσ e^{−mσ} = 1/m`; the `2^{−n_C}` factor converts between the two. The tree check (no loops, one `C` edge between two leaves, `n_C = 1`, prefactor `2T`): `2^{−1} · 2T · (1/m) e^{−mτ} = (T/m) e^{−mτ} = C₀` — exactly the tree correlator. This `2^{−n_C}` is applied in `diagram_value` / `diagram_value_x` (`full_integrator.py:1153`, `:1210`).

### 2.7 The causal-time integral: ordering chambers

What remains is `∫ ∏_v dt_v 𝟙(θ's) (smooth integrand)`. The retarded edges impose orderings `t_v > t_u` on the internal vertices — a **partial order (poset)**. A poset's domain decomposes into **ordering chambers**, one per **linear extension** (total order consistent with the partial order). Within each chamber every `|Δt|` has a fixed sign, so the integrand is smooth (no cusps) and ordinary quadrature converges fast (`causal_chambers.py:18-24`). This is the **whole point of doing momentum first**: the temporal pipeline integrates *products of exponentials* per chamber and hits a `1/β` close-pair pathology; here the momenta are already gone, so close-pair *cannot arise*.

`causal_chambers.causal_chambers(n, retarded_edges)` returns the list of linear extensions; an empty edge set gives all `n!` orderings (they tile the time cube). The poset machinery is **reused verbatim** from the temporal pipeline (`_CausalPoset`, `_enumerate_linear_extensions`).

### 2.8 The analytic spatial inverse Fourier transform (IFT)

Historically the code FT'd `MomFactor(q)` on a `q`-grid then numerically transformed `q → x`. The current production path avoids the `q`-grid entirely. Because each Schwinger sample's `q`-dependence is a Gaussian `pref · e^{−q⃗ᵀ𝓑 q⃗}` with `𝓑 = D·Q_eff`, the inverse FT to position is itself a closed-form **heat kernel** (`full_integrator.py:99-118`). For `n_ext = 1` (`k = 2`):

```
   ∫ dᵈq/(2π)ᵈ e^{iq·x} pref · e^{−𝓑 q²} = pref · (4π𝓑)^{−d/2} · e^{−|x|²/4𝓑}.
```

For `n_ext ≥ 2` (`k ≥ 3`) it is the multivariate Gaussian with `𝓑` shared across spatial components by isotropy (`full_integrator.py:108-110`, implemented in `_heat_kernel_x_general`). The result: **no `q`-grid, no `q_cut`, no ringing** — exact. This was the change that took a KPZ `max_ell=2` evaluation from ~17 min to ~16.6 s.

### 2.9 Derivative (form-factor) vertices

A *derivative* interaction vertex (e.g. Model B's conserved `∇²(φ²)`, Burgers' `∂_x(φ²)`, KPZ's `(∂_xφ)²`) deposits a **momentum-space form factor** `F(ℓ, q)` on the loop integrand (because `Lap → −|k|²`, `∂_x → i k`). The form factor is a *polynomial* in the momenta, so the loop integral becomes `MomFactor · ⟨F⟩`, where `⟨F⟩` is the average of `F` over the loop-momentum Gaussian `ℓ ~ N(ℓ̄, Σ)` with `ℓ̄ = −Lam⁻¹ N q` and `Σ = (2D Lam)⁻¹`. Because `F` is polynomial this average is computed **exactly** by **Gauss–Hermite quadrature** (`_formfactor_average`). The `q → x` transform of the resulting polynomial-times-Gaussian is again closed form via heat-kernel moments (`_formfactor_average_x`). Phase 3 (`d ≥ 2` derivative vertices) is not implemented and routes to the numerical FT.

### 2.10 The radial-Bessel × angular Monte-Carlo backend

A third backend (`method='bessel'`) reparametrizes the causal-region point `(u_v = −t_v, σ_e) = λ · ŝ` with `ŝ` on the `(n−1)`-simplex (`n = n_V + n_C`). The Symanzik polynomials are *homogeneous* (`U → λ^L Û`, `F → λ^{L+1} F̂`, `W → λ Ŵ`), so the **radial `λ` integral is exactly a modified Bessel function**:

```
   ∫₀^∞ λ^P e^{−aλ − c/λ} dλ = 2 (c/a)^{(P+1)/2} K_{P+1}(2√(ac)),
   a = μ Ŵ,   c = |x|² Û / (4D F̂),   P = n − 1 − (L+1)d/2.
```

Only the smooth angular simplex is sampled (Dirichlet(1,…,1) with causal-poset rejection). The radial reduction does the `det Lam → 0` (degenerate-loop) direction analytically, which is what cures the infinite-variance pathology of pure Monte-Carlo for derivative vertices at `ℓ ≥ 2`. See `_diagram_bessel_xs` (`full_integrator.py:362`).

---

## 3. External tools used

This subsystem is deliberately light on heavy dependencies. The two genuine numerical workhorses are **NumPy** and **SciPy**; **SymPy** appears only in the routing step; **Sage** and **nauty** are upstream (enumeration) and not imported here. There is **no numba**, **no networkx** in these files.

### 3.1 NumPy (`import numpy as np`)

**What it is.** NumPy is the foundational array library for numerical Python: it provides the n-dimensional array type (`ndarray`), vectorized elementwise math, linear algebra (`np.linalg`), and a fast einstein-summation engine (`np.einsum`). Operations run in compiled C and (for the heavy linear algebra) call BLAS/LAPACK, which release the Python GIL.

**How THIS code uses it** — exhaustively, because NumPy *is* the integrator:

* **Batched tensor contractions via `np.einsum`.** The Symanzik matrices are built for an entire grid of Schwinger samples at once. E.g. in `_momentum_factor_batch`:
  ```python
  Q   = np.einsum('pe,ej,ek->pjk', w_batch, b, b)   # (P, n_ext, n_ext)   full_integrator.py:73
  Lam = np.einsum('pe,el,em->plm', w_batch, a, a)   # (P, L, L)           full_integrator.py:78
  N   = np.einsum('pe,el,ej->plj', w_batch, a, b)   # (P, L, n_ext)       full_integrator.py:79
  ```
  Here `p` indexes the sample, `e` the edge, `l`/`m` the loop momenta, `j`/`k` the external momenta. `np.einsum('pe,ej,ek->pjk', ...)` reads "sum over `e`, keep `p,j,k`" — it computes `Σ_e w_{pe} b_{ej} b_{ek}` for every sample `p` in one call.
* **Batched linear algebra.** `np.linalg.det(Lam)` computes `U` for the whole batch (`:80`); `np.linalg.solve(Lamok, Nok)` computes `Lam⁻¹N` (the Schur complement ingredient) for all non-degenerate samples at once (`:85`); `np.linalg.inv`, `np.linalg.cholesky` appear in the form-factor average.
* **Boolean masking for degeneracy.** `ok = U > u_floor` (`:81`) selects the samples where the loop matrix is non-degenerate; degenerate samples (a self-loop where `U → 0`) get zero. This appears everywhere (`good`, `okk`, `okF`).
* **Gauss quadrature node/weight generators.** NumPy ships the classical orthogonal-polynomial quadratures used here:
  * `np.polynomial.legendre.leggauss(n)` — Gauss–Legendre, for the internal-vertex time integrals and the Schwinger `σ` grid (`_gl_on`, `full_integrator.py:354`, `:522`).
  * `np.polynomial.hermite_e.hermegauss(gh_order)` — Gauss–Hermite with weight `e^{−x²/2}`, for averaging a polynomial form factor over the loop-momentum Gaussian (`:216`).
* **Mesh/grid construction.** `np.meshgrid(*([s_nodes] * n_C), indexing='ij')` builds the `n_C`-dimensional Cartesian product of Schwinger nodes (`:528`).
* **Random sampling (Monte-Carlo / Bessel backends).** `np.random.default_rng(seed)` creates a reproducible PRNG; `rng.standard_exponential(...)` draws exponentials for the Dirichlet simplex sampling (`:389-391`); `-np.log(rng.random(P))/mu` draws `Exp(μ)` time gaps for the MC importance sampler (`:571`).
* **Complex dtype.** Form-factor averages use `dtype=complex` because `∂_x → ik` makes an odd-derivative vertex's contribution complex per diagram (`:208`); the physical correlator is real and the imaginary parts cancel in the diagram sum.

### 3.2 SciPy (`from scipy import integrate`, `from scipy.special import kv, gamma`)

**What it is.** SciPy is the scientific-computing layer built on NumPy: numerical integration, optimization, special functions, etc.

**How THIS code uses it:**

* **`scipy.integrate.quad`** (`causal_chambers.py:34`, used in `integrate_chamber`, `:78`). `quad` is adaptive Gauss–Kronrod 1-D quadrature (`∫ f(x) dx` with an error estimate). The oracle/reference path `integrate_over_chambers` uses nested `quad` calls to do the chamber simplex integral. (The *production* path in `full_integrator.py` does NOT use `quad` — it uses fixed Gauss–Legendre product grids for vectorization.)
* **`scipy.special.kv(nu, z)`** — the modified Bessel function of the second kind `K_ν(z)`. Used in `_diagram_bessel_xs` (`full_integrator.py:380`, `:464`) for the exact radial `λ`-integral `∫₀^∞ λ^P e^{−aλ−c/λ} dλ = 2(c/a)^{(P+1)/2} K_{P+1}(2√(ac))`.
* **`scipy.special.gamma(z)`** — the Gamma function `Γ(z)`. Used at `x = 0` (equal-point) where the Bessel integral reduces to `Γ(P+1)/a^{P+1}` (`full_integrator.py:458`).

### 3.3 SymPy (`import sympy as sp`)

**What it is.** SymPy is the pure-Python **computer-algebra system** (symbolic math): it represents symbols (`sp.Symbol`), exact rational/integer arithmetic, polynomial manipulation, and — crucially here — solves *linear systems symbolically*.

**How THIS code uses it** — only in `momentum_routing.py`, to solve momentum conservation once per diagram:

* `q = sp.symbols(f'q0:{k}')` — create the external-momentum symbols `q0 … q_{k-1}` (`momentum_routing.py:115`).
* `kk = {e: sp.Symbol(f'k{i}') ...}` — one unknown momentum symbol per edge (`:116`).
* Build the conservation **residuals** (one per vertex): `s += kk[e]` for an incoming edge, `s -= kk[e]` for an outgoing edge, minus the injected external momentum at a leaf (`:120-130`). `sp.expand` normalizes each residual.
* `sp.linsolve(residuals, list(kk.values()))` — SymPy's **linear-system solver**: solve the conservation equations for the edge momenta. It returns a solution set; an empty set means an inconsistent system (raised as an error — should never happen for a valid MSR diagram) (`:138-143`).
* The **free symbols** that SymPy leaves unpinned in the solution *are* the loop momenta; the code collects them (`expr.free_symbols`), subtracts the external symbols, sorts by name, and relabels to `ℓ_0 … ℓ_{L-1}` (`:149-158`).
* `RoutingResult.edge_coeffs()` extracts integer coefficients: `mm.coeff(s)` reads the coefficient of symbol `s` in the expanded edge momentum, and `_num` coerces it to a Python `int` (or `float`, or — defensively — leaves it symbolic) (`momentum_routing.py:79-95`). `sp.nsimplify` is used to recognize a number that is not already flagged as numeric.

The point of using a CAS here is robustness: momentum conservation is a small exact linear-algebra problem and SymPy gives integer `±1`/`0` routing coefficients with no floating-point ambiguity. SymPy is also used for *symbolic* `k_e²` in `RoutingResult.edge_k2()` (`:57-60`), though the integrator consumes `edge_coeffs()` (the signed linear coefficients), not `edge_k2()` (which squares and loses the signed cross terms — noted explicitly at `momentum_routing.py:72-74`).

### 3.4 Sage / nauty — NOT imported here (upstream)

The reader should know these exist but should *not* look for them in this subsystem.

* **SageMath** is a large open-source mathematics system (it bundles many libraries behind a Python interface). In this project it is used *upstream* for graph manipulation and the symmetry-factor computation `𝒮(Γ)`, and the project runs under `sage -python`. The `prediagram` object the descriptor reads (`td.prediagram[0]` is a graph `D`) is a Sage graph; `D.edges()` and `D.vertices()` are Sage methods. But this subsystem never calls Sage directly — it only consumes the already-built typed diagram.
* **nauty** ("No AUTomorphisms, Yes?") is a C library for graph canonicalization and automorphism groups, used *upstream* during diagram enumeration to dedup isomorphic diagrams. It is invisible here.

### 3.5 Python standard library

* `math` — scalar `math.pi`, `math.exp`, `math.factorial`, `math.comb` for closed-form prefactors (the batched paths use NumPy equivalents).
* `itertools` — `permutations` (the `∂ ln U` / `∂ 𝓑` derivative chains in `_dlnU_block` / `_dB_block`) and `product` (Wick-mapping enumeration). Imported lazily inside functions.
* `dataclasses.dataclass` — declares the descriptor record types `CEdge`, `CStackDiagram`, `RoutingResult` (see §4).

---

## 4. Components

Exhaustive, file by file. Signatures are reproduced verbatim. "Sample" = one point in the Schwinger-weight grid; `P` is the batch size; `E` edges; `L` loops; `n_ext = k − 1` external momenta; `n_x` output positions.

### 4.1 `diagram_descriptor.py`

#### `class CEdge` — `diagram_descriptor.py:46`
```python
@dataclass(frozen=True)
class CEdge:
    a: tuple        # loop-momentum routing coefficients (over the L loop momenta)
    b: tuple        # external-q routing coefficients (over the n_ext external momenta)
    kind: str       # 'R' (retarded segment) or 'C' (correlation line)
    u: int          # endpoint vertex id
    v: int          # endpoint vertex id  (u == v ⇒ self-loop = tadpole)
    external: bool  # True iff the edge touches an external leaf
    fpairs: tuple = ()   # coupled-field propagator matrix indices (Dyson 3c)
```
One C-stack edge. `couples_loop()` (`:68`) returns `True` iff any `a_i ≠ 0`, i.e. the edge participates in `∫dᵈℓ`; external legs have `a = 0` (they carry only `±q`). `fpairs` holds the underlying `G_R` half-edge propagator matrix indices `(resp_idx, phys_idx)` for the coupled-field path — for an `R` edge `((ri, pi),)`, for a `C` edge `((ri_u, pi_u), (ri_v, pi_v))`; it is an empty tuple for hand-built descriptors and the single-field path never reads it (`:54-66`).

#### `class CStackDiagram` — `diagram_descriptor.py:74`
```python
@dataclass(frozen=True)
class CStackDiagram:
    internal_vertices: tuple   # interaction-vertex ids (their times are integrated)
    external_legs: tuple       # leaf vertex ids, in correlation-function order
    edges: tuple               # tuple of CEdge
    n_loops: int               # == diagram loop_number
```
The C-stack view of a typed diagram — the canonical object every diagram is reduced to. `loop_edges()` (`:89`) returns the non-external (internal self-energy) edges. `is_tadpole_like()` (`:93`) returns `True` iff some edge is a self-loop (`u == v`) — *diagnostic only*; the production evaluator does not branch on it.

#### `_is_two_point_noise_source(asg)` — `diagram_descriptor.py:99`
Returns `True` iff `asg` is a `SourceType` with exactly two response legs and no physical legs — the Gaussian 2-point noise insertion `⟨φ̃φ̃⟩` that represents a `C` line.

#### `diagram_to_cstack(td) -> CStackDiagram` — `diagram_descriptor.py:105`
**The one mapping every diagram goes through.** Takes a `TypedDiagram` `td`; returns its `CStackDiagram`. Steps:
1. Unpack `td.prediagram` → graph `D`, leaves, internal vertices; read `td.vertex_assignments` (`:112-114`).
2. **Classify each non-leaf vertex** (`:116-142`): a `VertexType` → an interaction vertex (integrated time); a 2-point `SourceType` → a noise vertex (to be contracted into a `C` line); a `SourceType` with `≥ 3` response legs (non-Gaussian noise) → kept as an internal vertex with `n` outgoing `R` edges (the paper's all-retarded formulation; its time is integrated like any vertex). A `SourceType` with `< 2` response legs raises `NotImplementedError`. An unrecognized assignment raises.
3. `route_momenta(td)` → `RoutingResult`; `ec = rr.edge_coeffs()` → `{edge: (a, b)}` (`:144-145`).
4. Build an **incidence map** vertex → incident edges (`:148-152`).
5. **Contract each 2-point noise source** (`:157-175`): its two incident `G_R` edges become **one `C` edge** between the two *other* endpoints (the noise-source time integrated out analytically gives `C(Δt)`). A noise source whose two edges land on the *same* vertex → a `C` **self-loop** (`u == v`) — the tadpole. The `C` edge takes either half's routing coefficients (sign-invariant downstream, `:29-32`). Degree ≠ 2 raises.
6. **Every remaining `G_R` edge → an `R` edge** (`:178-187`). External legs (touching a leaf) are flagged `external=True`.
7. Assemble and return the `CStackDiagram` with internal vertices sorted, external legs in correlation-function order, and `n_loops = rr.n_loops` (`:189-193`).

**Key invariant the descriptor encodes** (`:10-13`): there is *no* bubble/tadpole/sunset branch. Whether a loop momentum couples to `q` (bubble vs tadpole) is a property of the Symanzik `F` *downstream*, not of this descriptor.

### 4.2 `momentum_routing.py`

#### `class RoutingResult` — `momentum_routing.py:50`
```python
@dataclass
class RoutingResult:
    edge_momenta: dict   # (u,v,lbl) -> sympy expr in q_syms + loop_syms
    q_syms: tuple        # external-momentum symbols q_0 … q_{k-1}
    loop_syms: tuple     # loop-momentum symbols ℓ_0 … ℓ_{L-1}
    n_loops: int
```
* `edge_k2()` (`:57`) → `{edge: sp.expand(k_e**2)}` — the symbolic squared momentum the heat-kernel pole needs. **Not used by the integrator** (squaring loses the signed cross terms; explicitly warned `:72-74`).
* `edge_coeffs()` (`:62`) → `{edge: (a, b)}` with `a` a tuple over `loop_syms`, `b` a tuple over `q_syms`, each coefficient coerced to Python `int`/`float` by the nested `_num` (`:79-89`). **This is what the Symanzik step consumes.**

#### `route_momenta(typed_diagram, verbose=False) -> RoutingResult` — `momentum_routing.py:99`
Solves momentum conservation over the diagram graph (full description in §3.3). Convention: edge `(u, v, lbl)` is oriented `u → v` carrying momentum `k_e` from `u` to `v`; `+` for an edge into a vertex, `−` for an edge out; at a leaf the injected external momentum `q_j` is subtracted. Overall conservation eliminates `q_{k-1}`. Returns the `RoutingResult` with relabeled loop symbols.

### 4.3 `spatial_reduce.py` (the reference single-sample Symanzik core)

#### `symanzik_matrices(a_list, b_list, weights)` — `spatial_reduce.py:58`
Builds `(Lam, N, Q)` for *one* set of edge weights. `a_list` is `E×L`, `b_list` is `E×n_ext`, `weights` is length `E`. Computes `aw = a * w[:,None]`, then `Lam = awᵀ@a`, `N = awᵀ@b`, `Q = (b*w[:,None])ᵀ@b`. Shapes `(L,L)`, `(L,n_ext)`, `(n_ext,n_ext)`. Raises on shape mismatch.

#### `symanzik_polynomials(a_list, b_list, weights)` — `spatial_reduce.py:82`
Returns `(U, Q_eff)` with `U = det Lam` (a float) and `Q_eff = Q − Nᵀ Lam⁻¹ N` (computed via `np.linalg.solve`, not an explicit inverse). Raises if `U ≤ 0` ("no loop-momentum damping — all Schwinger weights zero?"). For `n_ext == 1` the single `Q_eff` entry is the scalar `F_reduced`.

#### `momentum_integral(a_list, b_list, weights, q, D, spatial_dim=1)` — `spatial_reduce.py:101`
Evaluates `I_mom = (4πD)^{−Ld/2} U^{−d/2} exp[−D qᵀ Q_eff q]` for one sample. `q` may be a scalar (`n_ext==1`) or a length-`n_ext` sequence. Raises if `L == 0` (needs ≥ 1 loop) or if `q`'s length ≠ `n_ext`. This is the **textbook reference**; the production code re-implements it batched (next section).

### 4.4 `full_integrator.py` (the production integrator)

#### `_momentum_factor_batch(a, b, w_batch, q_vec, D, spatial_dim, u_floor=1e-300, return_gaussian=False)` — `full_integrator.py:56`
**The batched `MomFactor(w,q)`.** `a` is `E×L`, `b` is `E×n_ext`, `w_batch` is `P×E`. Returns `(P,)`. For `L = 0` (tree) → `exp(−D qᵀ Q q)`. For `L ≥ 1` → `(4πD)^{−Ld/2} U^{−d/2} exp(−D qᵀ Q_eff q)` with batched `det`/`solve` and a degeneracy mask `ok = U > u_floor`. If `return_gaussian=True` it also returns `(Lam, N, ok)` so a derivative vertex's form factor can be averaged over the loop Gaussian `ℓ ~ N(−Lam⁻¹Nq, (2D Lam)⁻¹)`. `Lam, N` are `None` for `L = 0`.

#### `_symanzik_kernel_batch(a, b, w_batch, D, spatial_dim, u_floor=1e-300, return_gaussian=False)` — `full_integrator.py:94`
The analytic-IFT sibling of the above: returns `(pref, Bcal, ok)` — the `q`-Gaussian `pref · e^{−q⃗ᵀ𝓑q⃗}` *uncollapsed* from `q`, with `pref = (4πD)^{−Ld/2} U^{−d/2}` and `𝓑 = D·Q_eff`. For `n_ext = 1`, `Bcal` is scalar `(P,)`; for `n_ext ≥ 2`, `Bcal` is the matrix `(P, n, n)`. `L = 0` (tree) → `pref = 1`, `𝓑 = D·Q`. With `return_gaussian` also returns `(Lam, N, Q)` for the derivative-vertex form-factor average. This is the function the heat-kernel IFT path is built on (no `q` contraction — the `x`-dependence stays analytic).

#### `_heat_kernel_x_general(Bcal_g, xs_arr, spatial_dim)` — `full_integrator.py:143`
The analytic spatial IFT of the per-sample `q`-Gaussian → the (multivariate) heat kernel, at evaluation points `xs_arr`. Returns `(P', n_x)`. For `n_ext = 1` it is `(4π𝓑)^{−d/2} e^{−|x|²/4𝓑}`; for `n_ext ≥ 2` it is the multivariate Gaussian `(4π)^{−nd/2} det(𝓑)^{−d/2} exp(−¼ Σ_c X⃗_cᵀ 𝓑⁻¹ X⃗_c)` (product over the `d` spatial components, which share `𝓑` by isotropy). The conjugate variables `X_j = x_{leg_j} − x_{leg_{k−1}}` (momentum conservation eliminated the last leaf). **Callers must pre-filter degenerate samples** (`det 𝓑 > 0`).

#### `_formfactor_average(formfactor, Lam, N, q_vec, D, ok, gh_order=6, spatial_dim=1)` — `full_integrator.py:187`
`⟨F(ℓ,q)⟩` of a derivative-vertex form factor over the loop-momentum Gaussian, by **Gauss–Hermite — EXACT for a polynomial `F`**. Mean `ℓ̄ = −Lam⁻¹N q`, covariance `Σ = (2D Lam)⁻¹` (Cholesky-factored). Generic in `L`, `n_ext`, and `d`. For `d = 1` it builds an `L`-dimensional GH grid; for `d ≥ 2` an `L·d`-dimensional grid (`q` placed on axis 0, the parallel component shifted by `ℓ̄`, the transverse components zero-mean). Returns `(P,)` complex (`∂_x → ik`), `1.0` where the loop is degenerate. Validated to 1e-12 vs a brute `∫dℓ`.

#### `_formfactor_average_x(formfactor, Lam, N, Q, D, ok, xs, spatial_dim=1, gh_order=6, q_deg=8)` — `full_integrator.py:253`
Phase 2 — analytic `q → x` IFT of a derivative-vertex form factor (`d = 1`, `k = 2`). Returns `(P, n_x)`. Two routes: (a) if the form factor exposes a `moment_x` callable, the joint-`(ℓ,q)`-Gaussian moment is used directly (one pass, no `q`-node loop); (b) otherwise the **polynomial-fit route** — interpolate `P(q) = ⟨F⟩_ℓ` at `q_deg+1` Chebyshev-like nodes (well-conditioned Vandermonde in a scaled variable `t = q/qsc`), then transform analytically via closed-form heat-kernel `q`-moments `E[(u + ix/2𝓑)^n]`, `u ~ N(0, 1/2𝓑)`. Raises for `spatial_dim ≠ 1` (Phase 3). This replaces the `n_q`-point numerical FT with `q_deg+1` GH evaluations — exact, no ringing.

#### `external_times_2pt(descr, tau)` — `full_integrator.py:341`
For a 2-point correlator: returns `{leaf0: 0.0, leaf1: τ}`. Raises if there are not exactly 2 leaves.

#### `_gl_on(lower, upper, n)` — `full_integrator.py:349`
Gauss–Legendre nodes/weights on `[lower, upper]` (broadcastable arrays), **√-concentrated toward `upper`** (where the retarded integrand peaks): `t = upper − (upper−lower)·v²`, `v ∈ [0,1]`. Returns `(t_nodes, t_w)` of shape `lower.shape + (n,)`. The `v²` substitution clusters nodes near the upper bound and carries the Jacobian `span·(wg·v)`.

#### `_diagram_bessel_xs(a, b, edges, internal, idx, internal_R, external_times, xs, mu, D, spatial_dim, formfactor, N, seed)` — `full_integrator.py:362`
The `method='bessel'` backend (full math in §2.10). Dirichlet-samples the angular simplex with causal-poset rejection (`valid &= (dd >= 0.0)` for internal `R` edges, `:405`), computes the hatted Symanzik invariants `Û, F̂, Ŵ`, and sums modified-Bessel terms `K_{P+1}(2√(ac))` (plain → single term; derivative → `Σ_m EF_m · K(P−m)` via the form factor's `moment_bessel`). `x = 0` keeps only the convergent `Γ`-function part (the equal-point is UV-sensitive). Returns `(n_x,)` real. Raises a clear `NotImplementedError` for `d ≥ 2` derivative vertices (no `moment_bessel`).

#### `diagram_kinematic(descr, q_vec, external_times, mu, D, spatial_dim=1, W=None, n_t=22, n_s=24, formfactor=None, gh_order=6, xs=None, method='grid', mc_n=1000000, mc_seed=0)` — `full_integrator.py:469`
**The heart.** The kinematic (unit-amplitude) full-diagram integral `∫ ∏dt_v ∏dσ_e 𝟙(θ) e^{−μΣw} MomFactor`, by causal-chamber quadrature. Walkthrough:
1. Extract `edges`, `internal` vertices, `a`/`b` coefficient matrices, `n_C` (`:487-493`).
2. Set the time window `[lo, hi]` from the external times and `W = 22/μ` (`:495-499`).
3. Build the **retarded poset on internal vertices**: internal→internal `R` edges become poset edges `internal_R`; `R` edges to/from a leaf become fixed-time scalar bounds `s_up`/`s_lo` (`:503-515`).
4. Build the **correlation Schwinger grid**: `σ = s_cap·v²` (`s_cap = 32/μ`) so nodes concentrate near `σ = 0` to resolve the integrable `U^{−d/2} ∼ σ^{−d/2}` self-loop singularity; the `e^{−μσ}` weight is folded into `s_w` (`:517-536`).
5. **Dispatch on `method`**: `'bessel'` → `_diagram_bessel_xs`; otherwise enumerate `chambers = causal_chambers(n_V, internal_R)` (`:553`).
6. For each chamber, build the **internal-time grid** — `'mc'` importance-samples via nested `Exp(μ)` gaps (`:564-578`); `'grid'` (default) builds the nested causal-chamber product grid latest→earliest, each level bounded above by the next-later time and its external scalar bound, via `_gl_on` (`:586-603`).
7. Compute the **edge weights** `w_batch` and the residual mass `mu_resid` (`R`: `tv − tu`; `C`: `|tu − tv| + σ`) and the per-sample amplitude `amp` (`:606-623`).
8. **Accumulate the result**, four sub-cases:
   * `xs given` + plain → Phase 1 analytic heat kernel (`:626-635`);
   * `xs given` + derivative, `n_ext ≥ 2` → multivariate Wick-moment IFT via `formfactor.moment_x_multi` (`:639-669`);
   * `xs given` + derivative, `n_ext = 1` → `_formfactor_average_x` (`:670-683`);
   * `q-eval` (no `xs`) → `_momentum_factor_batch` (× `_formfactor_average` for a derivative vertex) summed against `amp` (`:685-709`).
9. Return real for the analytic-IFT path; complex for a derivative vertex; float otherwise (`:714-716`).

Returns a float (or `(n_x,)` real for the IFT path). Grid size `≈ n_t^{n_V}·n_s^{n_C}` per chamber.

#### `spectral_rows(descr)` — `full_integrator.py:719`
The expanded **row** list for the coupled-field (unequal-`D`, Dyson 3c) kinematic. Each `R` edge → one retarded row `'R'`; each `C` edge → **two glued retarded segments** `'Cu'`/`'Cv'` sharing one Schwinger `σ` (the noise-source time integrated out). Returns `[(edge_index, edge, half)]`. The two-segment convention reproduces `2^{−1}` per `C` edge so a coupled diagram is `pv·Σ W·I_spec` with no separate `2^{−n_C}` (full derivation `:719-747`).

#### `_set_partitions(items)` — `full_integrator.py:750`
Generator of all set partitions of a list (Bell(n) of them) — used by the `(−|k|²)^n` momentum-insertion expansion and the Faà-di-Bruno chains.

#### `_dlnU_block(rows, G)` — `full_integrator.py:765`
`∂^{rows} ln U` for a multiset of inserted rows: `(−1)^{m−1} Σ_{cyclic orders} ∏ g` (closed `g`-chains), from `∂_t Λ = a_t a_tᵀ`. `G[(r,s)]` are per-sample arrays.

#### `_dB_block(rows, G, V, D)` — `full_integrator.py:784`
`∂^{rows} 𝓑` for a multiset of rows: `(−1)^{m−1} D Σ_{orderings} v_{p₁}(∏ g)v_{p_m}` (open `v–g–v` chains), from `∂_t v_r = −g_rt v_t`.

#### `diagram_kinematic_spectral(descr, q_vec, external_times, mass_table, D, spatial_dim=1, W=None, n_t=22, n_s=24, xs=None, mu_scale=None, power_table=None, insert_row=None, insert_rows=None)` — `full_integrator.py:802`
The coupled-field kinematic with **per-segment masses** (Dyson 3c). Same causal-chamber quadrature and Symanzik reduction as `diagram_kinematic`, but the mass factor is `∏_rows e^{−m_r w_r}` over `spectral_rows`, evaluated for a **batch of mass assignments** (`mass_table`, shape `(n_rows, n_assign)`) against ONE shared quadrature/Symanzik pass — the `N_modes^{labels}` spectral sum costs almost nothing beyond a single-mass integral. `power_table` adds the confluent `∏ w_r^{κ_r}` (equal-eigenvalue Duhamel strings). `insert_rows` multiplies the momentum integral by `(−|k_r|²)` per listed row, expanded over set partitions (`_ins_pieces`/`_ins_factor`, `:999-1044`) — the B26 `(−|k|²)^n` loop dressing. Plain vertices, deterministic grid only. `xs=None` → `(n_assign,)` complex at `q_vec`; `xs` given → `(n_assign, n_x)` complex IFT. Raises if any segment mass has `Re m ≤ 0`.

#### `diagram_value(descr, prefactor_val, q_vec, external_times, mu, D, spatial_dim=1, **kw)` — `full_integrator.py:1142`
One diagram's contribution to the cumulant: `2^{−n_C}·prefactor·kinematic`. The `2^{−n_C}` converts the `2T` noise convention to the unit-amplitude `C` edges.

#### `_is_retarded_type(descr)` — `full_integrator.py:1156`
`True` iff the two external legs have *different* propagator kinds (one `C`, one `R`) — a **retarded** self-energy insertion, which dresses both retarded and advanced sides → the pair `Σ_R(τ) + Σ_A(τ) = Γ(τ) + Γ(−τ)`. A `{R,R}` (Keldysh) insertion is its own conjugate → a single `Γ(τ)`.

#### `diagram_correlator(descr, prefactor_val, q, tau, mu, D, spatial_dim=1, **kw)` — `full_integrator.py:1166`
One diagram's contribution to `C(q,τ)` with the retarded+advanced sum applied: `Γ(τ)+Γ(−τ)` for a retarded-type insertion, else `Γ(τ)` (or `2Γ(0)` at `τ=0`).

#### `correlator_2pt(descrs_prefactors, q, tau, mu, D, spatial_dim=1, **kw)` — `full_integrator.py:1182`
The connected 2-point cumulant `C(q,τ) = Σ_Γ Γ(q,τ)` — sum over all enumerated diagrams (tree + every loop), each via the same full integral, dead diagrams (`|pre| < 1e-14`) skipped. Returns momentum-space `C(q,τ)`.

#### `diagram_value_x` / `diagram_correlator_x` / `correlator_2pt_x` — `full_integrator.py:1203` / `:1213` / `:1228`
The **analytic-IFT analogues** of the three functions above — they return real-space `δC(x,τ)` (a vector over `xs`) directly via the heat-kernel IFT (no `q`-grid, no numerical FT). `diagram_value_x` applies `2^{−n_C}·prefactor`; `diagram_correlator_x` applies the ret+adv sum; `correlator_2pt_x` sums over diagrams. **These are the production entry points** the bridge calls for real-space output.

#### `field_respecting_mappings(point_fields, leaf_fields)` — `full_integrator.py:1243`
All bijections canonical-point-slot → leaf-position that respect the field type, as tuples `m` (`m[j]` = slot assigned to leaf `j`). Each mapping is one external Wick contraction; the caller divides by `external_wick_compensation` (orbit–stabilizer). Mirrors the temporal `_all_mappings` enumeration validated to machine precision against the Boltzmann series at `k ≤ 5`.

#### `diagram_correlator_pts(descr, prefactor_val, x_pts, t_pts, mu, D, spatial_dim=1, mappings=None, comp=1, **kw)` — `full_integrator.py:1284`
One diagram's contribution to the **`k`-point** cumulant at general external events — the `k`-generic replacement for the 2-point `Γ(τ)+Γ(−τ)` completion. `x_pts` is `(n_pts, k[, d])`, `t_pts` is `(k,)`. For each Wick mapping it runs the kinematic with leaf times `t_pts[m[j]]` and IFT conjugates `X_j = x(m[j]) − x(m[k−1])`. It groups mappings that produce identical configurations and evaluates each once (a pure speed optimization). Returns `(n_pts,)` real, scaled by `2^{−n_C}·prefactor/comp`.

### 4.5 `causal_chambers.py`

#### `causal_chambers(n_vertices, retarded_edges)` — `causal_chambers.py:41`
The ordering chambers of the internal-vertex-time poset — **reuses** the temporal `_CausalPoset` + `_enumerate_linear_extensions`. `retarded_edges` is an iterable of `(u, v)` meaning `t_v > t_u`. Returns the list of linear extensions, each a length-`n_vertices` tuple (earliest vertex first). Empty edge set → all `n!` orderings.

#### `integrate_chamber(f, order, lo, hi, limit=80)` — `causal_chambers.py:58`
`∫ f(t) dt` over the chamber simplex `{lo ≤ t_{order[0]} ≤ … ≤ t_{order[-1]} ≤ hi}` by **nested `scipy.integrate.quad`**. `f` takes a length-`n` array of vertex times (indexed by vertex, not by order) and must be smooth (the C0/C1 momentum integral guarantees this). *Reference/oracle path* — the production `diagram_kinematic` uses fixed product grids instead.

#### `integrate_over_chambers(f, n_vertices, retarded_edges, lo, hi, limit=80)` — `causal_chambers.py:84`
The C2-full internal-time integral: **Σ over causal chambers** of `integrate_chamber`. Equals `∫_{[lo,hi]^n} f(t)·𝟙(retarded orderings) dt`.

### 4.6 `generic_evaluator.py` — ORACLE ONLY (not on the production path)

Banner-annotated `⚠ ORACLE-ONLY` (`generic_evaluator.py:2-4`): superseded by `full_integrator.py`, reached only by its own tests, kept as an independent numerical cross-check. `compute_cumulants` does NOT use it. It evaluates one diagram via the older **Dyson-convolution** route (self-energy `σ(q,u)` → single-mode Dyson dressing). Its functions, briefly:

* `loop_self_energy(descr, q, t, mu, D, T=1.0, spatial_dim=1, **quad)` (`:43`) — amputated self-energy of a 2-vertex bubble via `temporal_integrate.sigma_parametric`.
* `sigma_grid_direct(descr, q, u_grid, mu, D, spatial_dim=1, n_l=2600, L_cut=None, formfactor=None)` (`:65`) — vectorized kinematic self-energy on a whole `u`-grid by a direct `∫dᵈℓ`; ~100× faster than the parametric path.
* `_sigma_grid_axis`, `_dyson_retarded`, `_dyson_keldysh` (`:138`, `:147`, `:175`) — the adaptive convolution grid and the retarded/Keldysh Dyson kernels.
* `_kinematic_to_physical(descr)` (`:188`) — the `2^{−n_C}` factor (same convention as the production path).
* `bubble_delta_C`, `_phi2_zero`, `tadpole_delta_C`, `diagram_delta_C`, `delta_C_one_loop` (`:200`, `:232`, `:251`, `:303`, `:318`) — the bubble/tadpole branch and the 1-loop sum. Note these *do* branch on `is_tadpole_like()` (`:311`), unlike the production integrator. **This is the only place a topology branch survives — a sign of the older design.**

---

## 5. Data structures

| object | defined at | fields / shape | notes |
|---|---|---|---|
| `TypedDiagram` (input) | `msrjd/diagrams/type_assignment.py:30` | `.prediagram = (D, G, leaves, internal)`; `.vertex_assignments = {vid: VertexType\|SourceType}`; `.edge_types`; `.external_legs`; `.propagator_indices = {(u,v,lbl): (resp_idx, phys_idx)}` | `D` is a Sage graph; `D.edges()` yields `(u,v,lbl)` 3-tuples |
| `RoutingResult` | `momentum_routing.py:50` | `edge_momenta {(u,v,lbl): sympy}`, `q_syms` (len `k−1`), `loop_syms` (len `L`), `n_loops` | `.edge_coeffs()` → `{edge: (a,b)}` is what's consumed |
| `CEdge` | `diagram_descriptor.py:46` | `a` (tuple len `L`), `b` (tuple len `n_ext`), `kind ∈ {'R','C'}`, `u`, `v`, `external`, `fpairs` | frozen dataclass; `u==v` ⇒ tadpole self-loop |
| `CStackDiagram` | `diagram_descriptor.py:74` | `internal_vertices` (tuple), `external_legs` (tuple), `edges` (tuple of `CEdge`), `n_loops` | the canonical descriptor |
| `_CausalPoset` | `final_integral.py:692` | `m` (int), `edges` (tuple of `(u,v)` meaning `s_v>s_u`), `scalar_lowers`, `scalar_uppers` | reused from the temporal pipeline |
| `a` / `b` matrices | built in each kinematic fn | `a`: `(E, L)` float; `b`: `(E, n_ext)` float | from `[e.a for e in edges]` / `[e.b …]` |
| `w_batch` | inside `diagram_kinematic` | `(P, E)` float — one Schwinger weight per edge per sample | `P ≈ n_t^{n_V}·n_s^{n_C}` per chamber |
| `Lam, N, Q` | Symanzik batch | `(P,L,L)`, `(P,L,n_ext)`, `(P,n_ext,n_ext)` | `U = det Lam`, `Q_eff = Q − Nᵀ Lam⁻¹ N` |
| `Bcal` (`𝓑`) | `_symanzik_kernel_batch` | scalar `(P,)` for `n_ext=1`; matrix `(P,n,n)` for `n_ext≥2` | `𝓑 = D·Q_eff`; the heat-kernel width |
| `mass_table` (coupled) | `diagram_kinematic_spectral` | `(n_rows, n_assign)` complex | column `j` = per-row segment masses of assignment `j` |

### Notation crib (code ↔ paper, from `full_integrator.py:46-53` and `spatial_reduce.py:47-54`)

| code | symbol | meaning |
|---|---|---|
| `Lam` | `Λ` | loop / first-Symanzik matrix `Σ_e w_e a_e a_eᵀ`; `U = det Lam` |
| `Bcal` | `𝓑(w)` | external quadratic form `= D·Q_eff = Q − Nᵀ Lam⁻¹ N`, scaled by `D` |
| `N`, `Q` | `N_rb`, `Q_ab` | Symanzik cross / external blocks |
| `a`, `b` | `B_er`, `C_eb` | edge routing coefficients (plain `B`, `C` in the paper) |
| `D` | `D_0` | scalar reference diffusion |
| `Scal` / dict key `'M'` | `𝒮(Γ)` | symmetry factor |

---

## 6. Data flow

**In:** `pipeline_bridge.build_pipeline_records(...)` yields `(td, pre)` pairs per loop order, where `td` is a `TypedDiagram` and `pre` is a symbolic prefactor. The bridge substitutes numeric couplings → `pv = float(SR(pre).subs(...))`, builds the descriptor `dd = diagram_to_cstack(td)`, optionally a form factor `ff`, and tags the loop order `ell` (`pipeline_bridge.py:1564-1572`). Diagrams with `|pv| < 1e-14` are dropped ("dead at the saddle").

**Through:** for real-space output the bridge calls, per live diagram and per `τ`:
```python
dCx_by_ell[el][it, :] += diagram_correlator_x(
    dd, pv, xg, float(tau), mu0, D0, spatial_dim=d,
    n_t=nt, n_s=ns, formfactor=ff, ...)            # pipeline_bridge.py:1697
```
which flows `dd, pv → diagram_value_x → diagram_kinematic(..., xs=xg)` → the chamber loop → `_symanzik_kernel_batch` → `_heat_kernel_x_general` → accumulated `δC(x)`.

**Out:** `diagram_kinematic` returns a float (q-path) or `(n_x,)` real vector (IFT path); `diagram_value*` multiplies by `2^{−n_C}·pv`; `diagram_correlator*` adds the ret+adv completion; `correlator_2pt*` sums over diagrams. The bridge accumulates `dCx_by_ell[el]` of shape `(n_tau, n_x)`, which becomes `compute_cumulants`'s `δC(x,τ)`.

**Coupled (unequal-`D`) path:** `compute_coupled_loop_correlator` (`pipeline_bridge.py:1120`) builds a `mass_table` from the spectral projectors and feeds `diagram_kinematic_spectral` instead — the per-row segment masses run over `N_modes^{n_rows}` spectral assignments in one quadrature pass.

**Concrete shape example (a 1-loop bubble, `k=2`, `d=1`):** two internal vertices (`n_V = 2`), two correlation lines (`n_C = 2`), `L = 1`. `causal_chambers(2, internal_R)` returns the orderings of the 2 vertices. Per chamber the grid is `P ≈ 22^2·24^2` samples; `a` is `(E,1)`, `b` is `(E,1)`; `Lam` is `(P,1,1)`, `U = w₁+w₂`, `Q_eff = w₁w₂/(w₁+w₂)`; `Bcal = D·Q_eff` scalar; the heat kernel `(4π𝓑)^{−1/2} e^{−x²/4𝓑}` is summed over the grid into `δC(x)`.

---

## 7. Gotchas & caveats

* **macOS fork crash (project-wide).** Fork-based multiprocessing crashes the user's kernel and OS; the spatial path is serial / thread-only by mandate. None of these six files fork, but be aware when extending. (See repo memory.)
* **`edge_k2()` must NOT be used for the reduction.** `RoutingResult.edge_k2()` squares the momentum and loses the *signed* cross-term coefficients; only `edge_coeffs()` carries the `(a, b)` the Symanzik step needs (`momentum_routing.py:72-74`).
* **Edge sign is irrelevant to the Symanzik forms.** Flipping `(a,b) → (−a,−b)` leaves every `Σ_e w_e aa`, `Σ_e w_e ab`, `Σ_e w_e bb` invariant, so the `C`-edge contraction may take either half's coefficients (`diagram_descriptor.py:29-32`). This is *relied on* — do not "fix" it.
* **Degeneracy floors are everywhere.** `u_floor = 1e-300`, `Bcal > 1e-300`, `det 𝓑 > 1e-300` mask out samples where the loop matrix degenerates (e.g. a self-loop has `U = σ → 0`). Degenerate samples contribute **zero** — silently. A diagram whose entire grid is degenerate returns 0 with no warning.
* **`σ`-node concentration is load-bearing.** The substitution `σ = s_cap·v²` (not plain Gauss–Laguerre) concentrates nodes near `σ = 0` to resolve the integrable `U^{−d/2} ∼ σ^{−d/2}` self-loop singularity that plain Laguerre *under-resolves* (`full_integrator.py:517-525`). Changing this quadrature will quietly degrade tadpole accuracy.
* **The memory wall at `ℓ ≥ 2`.** The chamber grid is `P = n_t^{n_V}·n_s^{n_C}` *per chamber* — the curse of dimensionality. A KPZ 2-loop diagram (`n_V=4, n_C=3`) is `≈ 1.8e8` points/chamber → tens of GB. The bridge refuses up-front (`SpatialPropagatorError`, `pipeline_bridge.py:1650`) and points to: lower `max_ell`, `SPATIAL_INTEGRATOR=mc`, coarser `SPATIAL_GRID_NT/NS`, or raise `SPATIAL_MEM_BUDGET_GB`. The production grid is `'grid'`; `'mc'` and `'bessel'` are the bounded-memory escape hatches.
* **MC is biased for derivative vertices.** The Monte-Carlo backend is validated `<0.1%` for *plain* `φⁿ` vertices but is BIASED for derivative vertices (the `det Lam → 0` singularity gives infinite variance). Use `SPATIAL_INTEGRATOR=bessel` for derivative vertices at `ℓ ≥ 2` (the radial reduction cures it). Stated repeatedly (`full_integrator.py:560-563`, `pipeline_bridge.py:1625`).
* **`x = 0` (equal-point) is UV-sensitive.** In the Bessel backend the `x = 0` term is divergent term-by-term; only the convergent `Γ`-function part is kept (`full_integrator.py:454-458`). Equal-point correlators are inherently UV-sensitive in `d ≥ 2`.
* **`d ≥ 2` derivative vertices are NOT implemented (Phase 3).** `_formfactor_average_x` raises for `spatial_dim ≠ 1`; the Bessel backend raises if `moment_bessel` is absent; the bridge routes these to the numerical FT (`full_integrator.py:276-279`, `:437-442`).
* **Genuine drift is refused.** A constant advection `v·∂_xφ` with a nonzero propagator drift `V ≠ 0` at the saddle is validated only at the heat-kernel oracle level, *not* in the integrator; the bridge raises rather than silently dropping the drift (`pipeline_bridge.py:1548-1554`). Only `φ*=0` gradient theories (Burgers/KPZ, where `V → 0`) run end-to-end.
* **Complex per-diagram values.** A derivative vertex with an odd number of `∂`'s gives a complex per-diagram contribution (`∂_x → ik`); the *physical* correlator is real and the imaginary parts cancel in the diagram sum / are dropped at the real-space output (`full_integrator.py:710-716`). Do not panic at a complex intermediate.
* **The `causal_chambers` empty-edge case tiles the cube.** With no retarded ordering constraints, `causal_chambers` returns all `n!` orderings, which *partition* the time cube — the integral over the cube equals the sum over chambers. Forgetting this gives an `n!` over/undercount.
* **`generic_evaluator.py` is an oracle — do not call it from production.** It still branches on bubble/tadpole topology (`is_tadpole_like`), the opposite of the production design. It is kept only for cross-checking.
* **Coupled path requires `Re m > 0` for every segment.** `diagram_kinematic_spectral` raises if any spectral-assignment mass has non-positive real part (stability), and clamps `mu_scale` to the slowest decay (`full_integrator.py:888-895`).
* **Possible inconsistency — see open questions** regarding `mu` vs `mu_resid` in the `R`-to-leaf edge handling.

---

## 8. Glossary

* **MSR-JD** — Martin–Siggia–Rose–Janssen–De Dominicis: the path-integral formalism that turns a stochastic PDE into a field theory with a physical field `φ` and a response (hatted) field `φ̃`.
* **Heat-kernel / `(k,t)` representation** — propagators written in momentum–time space; `G_R(k,t) = θ(t)e^{−(μ+Dk²)t}`. The key property: edge times act as Schwinger parameters.
* **Schwinger parameter** — an auxiliary integration variable (here a time `w` or a noise-source `σ ∈ [0,∞)`) that linearizes a propagator denominator into an exponential, making the momentum integral Gaussian.
* **Symanzik polynomials** — `U` (first / "first graph polynomial", `= det Lam`, sum over spanning trees) and `F` / `Q_eff` (second, the Schur complement carrying external-momentum dependence). The Gaussian loop integral collapses to these.
* **Routing coefficients `(a_e, b_e)`** — the integer (`±1`/`0`) coefficients in `k_e = Σ a_{ei} ℓ_i + Σ b_{ej} q_j`; how loop and external momenta thread each edge.
* **Loop momentum `ℓ`** — a free momentum left undetermined by conservation; there are `L` (= the diagram's loop number) of them.
* **`R` edge / retarded line** — a `G_R` propagator segment; carries a causal `θ(w ≥ 0)`.
* **`C` edge / correlation line** — a `⟨φφ⟩` line; in this representation it is two `R` segments glued at a 2-point noise source, with one Schwinger parameter `σ`.
* **Tadpole** — a `C` self-loop (`u == v`): a decoupled `⟨φ²⟩`-type loop that does not carry external momentum.
* **Bubble** — a 1-loop self-energy with two internal vertices; its loop momentum couples to `q`.
* **Sunset** — a 2-loop diagram (three lines between two vertices, the canonical `L=2` test case).
* **Causal chamber** — one ordering region of the internal-vertex-time integral, corresponding to a **linear extension** of the retarded poset; within it every `|Δt|` sign is fixed so the integrand is smooth.
* **Poset / linear extension** — a partial order (here `t_v > t_u` from retarded edges) and a total order consistent with it (one chamber).
* **Symmetry factor `𝒮(Γ)`** — the combinatorial prefactor of a diagram (in the prefactor, computed upstream with Sage).
* **`2^{−n_C}`** — the convention factor converting the enumeration's `2T` noise-vertex normalization to the integrator's unit-amplitude `C` edges.
* **IFT (inverse Fourier transform)** — `q → x`; here done analytically because each Schwinger sample's `q`-dependence is Gaussian → a closed-form heat kernel.
* **Form factor** — the polynomial in momenta that a derivative interaction vertex deposits on the loop integrand (`Lap → −|k|²`, `∂_x → ik`).
* **Gauss–Legendre / Gauss–Hermite / Gauss–Laguerre quadrature** — fixed-node numerical integration rules exact for polynomials up to a degree; Legendre for finite intervals (vertex times, `σ`), Hermite for Gaussian-weighted integrals (form-factor loop average), Laguerre for `[0,∞)` with `e^{−x}` weight.
* **Schur complement** — `Q − Nᵀ Lam⁻¹ N`; the residual quadratic form after integrating out the `ℓ` block — exactly `Q_eff`.
* **`SourceType` / `VertexType`** — the typed-vertex classes (`msrjd/core/vertices.py`): a noise source (`response_legs` only) vs an interaction vertex (response + physical legs).
* **`fpairs` / `propagator_indices`** — `(resp_idx, phys_idx)` matrix indices identifying which field's propagator a half-edge uses, for the coupled multi-field path.
* **Spectral assignment** — for the coupled (unequal-`D`) path, a labeling of every segment with one of the `N_modes` spectral eigenmodes; the diagram sums over all such labelings.
* **`det Lam → 0` singularity** — the degenerate-loop direction (e.g. self-loops) where `U → 0`; the source of MC infinite variance, cured analytically by the radial-Bessel reduction.
* **SymPy `linsolve`** — SymPy's exact linear-system solver, used to route momenta.
* **`scipy.special.kv` / `K_ν`** — modified Bessel function of the second kind, the closed form of the radial `λ`-integral in the Bessel backend.

---

## 9. Proposed manual subsections

1. **Why momentum-first** — the close-pair pathology of the temporal pipeline and how doing the momentum integral analytically removes it.
2. **The heat-kernel representation** — `G_R`, `C₀`, and the `C`-line-as-two-`R`-segments identity.
3. **Momentum routing** — conservation at spatial vertices, the linear solve, and the routing coefficients `(a_e, b_e)`.
4. **Symanzik polynomials from scratch** — `Lam`, `N`, `Q`; `U = det Lam`; `Q_eff` as a Schur complement; the spanning-tree / graph-polynomial picture; the bubble and sunset worked by hand.
5. **The collapsed loop integral** — the Gaussian `∫dᵈℓ` → `(4πD)^{−Ld/2} U^{−d/2} e^{−DqᵀQ_eff q}`, exact in `L` and `d`.
6. **The descriptor** — `diagram_to_cstack`, noise-source contraction, and why there is no bubble/tadpole branch.
7. **Causal chambers** — posets, linear extensions, and smooth per-chamber quadrature.
8. **The kinematic integral** — `diagram_kinematic` walkthrough: the time window, the retarded poset bounds, the `σ` grid, the chamber loop.
9. **Normalization** — the `2^{−n_C}` factor and the tree check.
10. **From `q`-space to `x`-space** — the analytic heat-kernel IFT (`n_ext=1` and the multivariate `k≥3` case).
11. **Derivative (form-factor) vertices** — the Gauss–Hermite loop average, the polynomial-fit `q→x` route, and the `d≥2` Phase-3 limitation.
12. **Backends** — `grid` (default), `mc` (bounded memory, plain only), `bessel` (radial-Bessel × angular MC, derivative-vertex-safe).
13. **The coupled (unequal-`D`) path** — `spectral_rows`, `diagram_kinematic_spectral`, mass tables, and the Dyson `(−|k|²)^n` insertions.
14. **General `k`-point cumulants** — Wick mappings and `diagram_correlator_pts`.
15. **Limits, guards, and the memory wall** — what raises, what is silently zeroed, and the environment knobs.
16. **The oracle evaluator** — `generic_evaluator.py` as an independent cross-check (and why it still branches on topology).

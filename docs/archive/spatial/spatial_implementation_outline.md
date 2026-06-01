# Spatial field theories in Daedalus — implementation outline (v1)

**Target**: enable theories with continuous spatial coordinates
`φ(x, t)` and Laplacian (or similar) kinetic operators, computing
spatial correlators `⟨φ(0, 0) φ(x, τ)⟩` at tree and 1-loop, off-
critical (`μ > 0`), in d = 1 to start.

**Reference papers**: A. Andreanov, G. Biroli, J.-P. Bouchaud,
A. Lefèvre, Phys. Rev. E **74**, 030101(R) (2006); A. Lefèvre and
G. Biroli, J. Stat. Mech. (2007) P07024.  Both in `Literature/`.

## Design principles

1. **The user writes the action and MF equations explicitly, the
   same way they do for time-only theories.**  The framework gains
   one new symbol — `Laplacian(phi)` — that the user can use in
   action / equation expressions, in the same way they currently
   use `Dt`.  No auto-injection of diffusion or noise terms; the
   user is in full control of what their theory looks like.
2. **Boundary conditions and initial conditions are first-class
   declarations** added to the theory spec.  v1 supports infinite
   domain and periodic; ICs default to stationary (system at MF
   saddle).  Other BCs / ICs are explicit v1.5/v2 work.
3. **Integration in (t, x), not (ω, k).**  After symbolic
   construction of `G(ω, k)`, the framework inverts both
   transforms symbolically to obtain `G(t-t', x-x')` (heat
   kernel × exponential decay for the free Allen-Cahn case).  Loop
   integrations then run in (t, x) space using a direct extension
   of the framework's existing polytope-time-integration machinery
   in Phase J — each loop adds spatial-coordinate integration
   variables alongside the existing time variables.  This choice:

   * matches the framework's existing time-domain architecture
   * yields native real-space output `C(x, τ)` (no final FT
     post-processing)
   * makes PBC clean via image-source sums in `G(t, x)` (no
     Brillouin-zone discretization)
   * keeps the tree-level integrands closed-form (Gaussian
     convolutions of heat kernels)

   The trade-off is that (t, x) becomes less natural at v2+ for
   critical phenomena, RG / ε-expansion, or higher-derivative
   kinetic operators where the propagator loses its closed-form
   inverse FT.  Those are explicitly later-stage concerns.

## What changes from the time-only case

| User-facing feature | Status in spatial extension |
|---|---|
| Action text syntax | **Same**, gains `Laplacian(field)` as a recognized symbol |
| MF equation syntax | **Same**, gains `Laplacian` as usable symbol |
| `declare_cgf_term` / Noise tab | **Same** (spatial structure of noise white-in-space implicit; explicit spatial-correlation kernels are v1.5) |
| `physical_field` declaration | Gains `spatial_dim`, used to register `Laplacian` in the namespace and tell the propagator builder this field has spatial structure |
| `parameter` declaration | **Same** |
| **`.boundary('periodic', length='L')`** | **New** — declares the spatial geometry |
| **`.initial('stationary')`** | **New** — declares the initial-condition mode (only stationary supported in v1) |
| `compute_cumulants` | Gains `spatial_grid` parameter; output gains `C_tau_x` keyed array |
| Diagram enumeration / typing | **Same** (topology unchanged by spatial structure) |

What's actually new from the framework-internal perspective is
concentrated in three areas:

* **Namespace + action parsing**: recognize `Laplacian` as an
  inert SR symbol parallel to `Dt`
* **Propagator builder**: take an additional symbolic inverse FT
  in `k` to produce `G(t-t', x-x')`; handle PBC via image-source
  sums in space
* **Phase J integrator**: extend the polytope-time-integration
  machinery to integrate over internal *positions* alongside
  internal *times*; the integrand factors become heat-kernel-
  times-exponential instead of pure exponential

Diagram enumeration, typing, causality, symmetry, and the
diagram→Phase-J pipeline are unchanged.

## Test theory (v1 target)

```python
# theories/allen_cahn_1d_subcritical_infinite.theory.py
TheoryBuilder('1D stochastic Allen-Cahn (subcritical, infinite domain)',
              n_populations=0)
.physical_field('phi', spatial_dim=1, description='order parameter')
.parameter('mu',  default=1.0, domain='positive')
.parameter('D',   default=1.0, domain='positive')
.parameter('lam', default=0.1, domain='positive')
.parameter('T',   default=1.0, domain='positive')
.declare_cgf_term('noise', response_legs=['phit','phit'], order=2,
                  coefficient='2*T', kernel='dirac_delta(tau)')
.set_action_text('phit*((Dt + mu - D*Laplacian)*phi + lam*phi^3)')
.equation(lhs='(Dt + mu - D*Laplacian)*phi', rhs='-lam*phi^3')
.boundary('infinite')
.initial('stationary')
.build()
```

The PBC variant is identical except `.boundary('periodic',
length='L')` and an `L` parameter.

## Mathematical content of the (t, x) approach

For the free Allen-Cahn propagator, the symbolic computation
proceeds as

* `K(ω, k) = iω + μ + D k²`  (from `(Dt + μ - D Laplacian)` in the
  action's bilinear sector after FT)
* `G(ω, k) = 1 / (iω + μ + D k²)`
* Close ω-contour: `G(t-t', k) = e^{-(μ + D k²)(t-t')} θ(t-t')`
* Close k-contour (inverse FT in k, free-theory case): heat
  kernel × exponential decay,
$$G(t-t', x-x') = \frac{1}{\sqrt{4 \pi D (t-t')}}
\exp\!\left[-\frac{(x-x')^2}{4 D (t-t')} - \mu (t-t')\right] \theta(t-t')$$

For PBC of period `L`, image-source sum:
$$G_{\text{PBC}}(t-t', x-x') = \sum_{n=-\infty}^\infty G_{\text{inf}}(t-t', x-x'+nL)$$

For the 1-loop tadpole self-energy correction, the additional
loop integral is

$$\Sigma^{(1\text{-loop})}(t, x) = -3 \lambda \int dt_v \int dx_v
G_R(t-t_v, x-x_v) \cdot G(0, 0) \cdot G_R(t_v - t', x_v - x')$$

where the internal vertex coordinate `(t_v, x_v)` is integrated
over the polytope `t > t_v > t'` (causality) crossed with
unconstrained `x_v ∈ ℝ` (translation invariance in space).  The
"tadpole" factor `G(0, 0)` is the equal-time-equal-position
fluctuation, which is UV divergent in d ≥ 1 — at d = 1 it's a
finite renormalization that doesn't need cutoffs for the
subcritical theory.

## Stage breakdown

### Stage 0 — design decisions (1 week, no code)

Sign off on:

1. **Spatial-operator grammar**: `Laplacian(phi)` function-call
   syntax (parallels existing `phi[i](v[i])` style).
2. **Single ambient spatial dim per theory** (mixed-dim out of v1
   scope).
3. **BCs in v1**: `infinite` (default) and `periodic` only.
4. **ICs in v1**: `stationary` only.
5. **Output observables in v1**: real-space `C(x, τ)` only; users
   wanting `S(q, ω)` apply a final FFT in post-processing.
6. **Integration representation**: (t, x), per the analysis in
   the design-principles section.

### Stage 1 — Action namespace + parsing (1.5 weeks)

Register `Laplacian` as an inert SR symbol in the namespace when
any field declares `spatial_dim ≥ 1`.  Saddle-killer rule:
`Laplacian * <saddle> = 0` (spatially-uniform saddles).

**Touch points**:

* `pipeline/theory.py` — `physical_field(spatial_dim=...)`; FieldSpec
  carries spatial_dim; emit model['spatial'] block in build()
* `msrjd/core/field_theory.py:_build_namespace` — register
  `Laplacian`, `Grad`, `Div` SR variables when spatial fields
  declared
* `pipeline/theory_compiler.py:make_action_lambda` — add the
  saddle-killer rule for `Laplacian * <saddle>`

**Acceptance**: `ft.expand()` on the Allen-Cahn test theory runs
to completion, the (1,1) bilinear sector contains a
`xt·dx · (Dt + μ - D · Laplacian)` term, and the saddle equations
contain a `D · Laplacian * φ*` term that the DAE solver
recognizes as 0 at the uniform saddle.

### Stage 2 — Propagator builder, infinite domain (1.5 weeks)

Extend `pipeline/_propagator.py` to:

1. Recognize the `Laplacian` symbol in `_to_kernel`, route it to a
   new symbolic image variable (call it `z_lap`).
2. Build `K_ft(ω, k) = ... + D · k²` by substituting
   `z_lap → -k²` in the FT step alongside the existing
   `z_delta → δ(t)`, `z_delta_p → δ'(t)`.
3. Take an *additional* symbolic inverse FT in `k` to produce
   `G(t-t', x-x')` for the free theory.  For Allen-Cahn this is
   the closed-form heat kernel × exponential decay.

For the general (multi-field, multi-Laplacian) case, Sage's
`integrate` may not produce a clean closed form.  Initial
implementation: detect "Allen-Cahn-like" propagators
(diagonal in field index, scalar `D k²` term) and use the heat-
kernel closed form; otherwise fall back to numerical inverse FT
on a grid.

**Touch points**:

* `pipeline/_propagator.py:_to_kernel` — `Laplacian` decomposition
* `pipeline/_propagator.py:build_propagator` — k-substitution; new
  `_inverse_ft_to_spatial(...)` stage that produces
  `G_tx(t, x)` callable from `G_ft(ω, k)`
* New module `msrjd/integration/spatial/heat_kernel.py` — closed-
  form heat-kernel utilities and the fallback numerical inverse FT

**Acceptance**: `build_propagator(allen_cahn_model)['G_tx'](t, x,
mu=1, D=1)` returns the expected heat-kernel × exp(-μt) values to
numerical precision for several (t, x) sample points.

### Stage 3 — Periodic boundary conditions (1 week)

Image-source method for PBC:
`G_PBC(t, x) = Σ_n G_inf(t, x + nL)`.

**Touch points**:

* `pipeline/theory.py` — `.boundary('periodic', length=...)` API;
  serialize / round-trip
* `pipeline/_propagator.py:build_propagator` — when PBC declared,
  wrap `G_tx` in the image-source sum (truncate at the
  exponentially-decaying tail)
* Stage 2's heat-kernel module gains the image-sum utility

**Acceptance**: Free-theory equal-time variance under PBC matches
the lattice-sum closed form
`⟨φ²⟩_PBC = (T/L) Σ_n 1/(D(2πn/L)² + μ)` to ~1e-6 relative.

### Stage 4 — Initial-condition machinery (0.5 weeks)

`.initial('stationary')` declared and validated.  For
stationary, no additional action term is needed; the framework
just confirms this is what the user wants and proceeds with the
existing Phase J machinery.

The implementation is mostly hooks for v1.5 extension to
transient ICs.  v1's only material change: validate that any
requested observable is consistent with stationary IC (i.e.,
two-time correlators are fine; one-time-only observables like
`⟨ρ(x, 0)⟩` only make sense averaged over the IC ensemble, which
needs an explicit initial distribution).

**Touch points**:

* `pipeline/theory.py` — `.initial(mode, ...)` API
* `pipeline/theory_serialize.py` — round-trip
* `pipeline/compute.py` — validate observable / IC compatibility

**Acceptance**: stationary-IC validation passes for the test
theory; requesting a transient observable raises a clear "v1
supports stationary IC only" error.

### Stage 5 — Phase J extension to (t, x) (2 weeks)

The biggest stage by LOC.  Extend the existing time-domain
polytope integration to also integrate over internal
*positions*.

For tree-level diagrams (no loops): each vertex's position is
determined by the action's structure (vertex at the noise
correlator becomes the integration "anchor" via the existing
machinery).  The new step is the spatial Fourier transform of
the noise vertex factor against the spatial-domain propagator
legs — for white-in-space noise (`⟨η(x, t) η(x', t')⟩ ∝ δ(x -
x')`), this collapses the internal spatial coordinate, leaving
just the external-position structure of `G(t, x)`.

For 1-loop diagrams: each loop adds one internal vertex
position `x_v` integrated over `ℝ` (or the periodic cell for
PBC), in addition to the existing time-polytope coordinate `t_v`.
The integrand factor is the product of heat-kernel-style
propagators evaluated at the appropriate `(Δt, Δx)` pairs.  For
the Allen-Cahn 1-loop tadpole, the spatial integral is a
single Gaussian convolution — closed form via Sage's `integrate`.

**Touch points**:

* `msrjd/integration/time_domain/final_integral.py` — extend
  vertex-time bookkeeping (`vertex_leg_time`, etc.) to track
  vertex *positions* alongside vertex times; thread spatial
  coordinates through the integrand-builder
* New module `msrjd/integration/spatial/spatial_integral.py` —
  spatial-coordinate handling for the polytope path; spatial
  integral evaluators (closed-form Gaussian for d=1 free
  theory; quadrature fallback for higher d / interacting cases)
* `pipeline/_propagator.py` — propagate spatial-coordinate data
  through to the integrator

**Acceptance**: 1-loop self-energy correction to the
equal-time variance of Allen-Cahn at λ = 0.1 matches the
analytic 1-loop tadpole `-3 λ T² / (8 μ √(μ D))`.

### Stage 6 — Output API + test theory + validation suite (1 week)

`compute_cumulants` gains `spatial_grid` parameter; output gains
`C_tau_x` array of shape `(len(tau_grid), len(spatial_grid))`.

**Touch points**:

* `pipeline/compute.py` — `spatial_grid` parameter; build
  `(t, x)` external grid for the integrator; populate `C_tau_x`
  in the result dict
* `theories/allen_cahn_1d_subcritical_infinite.theory.py` — new
* `theories/allen_cahn_1d_subcritical_pbc.theory.py` — new
* `models/allen_cahn_1d_sim_numba.py` — new 1D Langevin simulator
  (spectral or finite-difference)
* `notebooks/pipeline_allen_cahn_1d_sim_compare.ipynb` — new
* `tests/test_spatial_rd_basics.py` — closed-form benchmarks

**Validation suite**:

| Test | Closed-form expectation |
|---|---|
| Free tree-level, infinite domain `⟨φ(0,0) φ(x,0)⟩` | `(T/(2√(μD))) · exp(-\|x\|/ξ)`, `ξ = √(D/μ)` |
| Free tree-level, infinite domain `⟨φ²⟩` | `T/(2√(μD))` |
| Free tree-level, PBC `⟨φ²⟩` | `(T/L) Σ_n 1/(D(2πn/L)² + μ)` |
| PBC → infinite limit (sweep L) | Result → `T/(2√(μD))` as `L → ∞` |
| 1-loop self-energy (φ⁴) | `-3λT² / (8μ√(μD)) + O(λ²)` |
| Simulator cross-check | 1D Langevin at λ = 0.1; framework matches within sim noise |

## Sequencing summary

| Stage | Effort | Cumulative |
|---|---|---|
| 0 — design decisions | 1 wk | 1 wk |
| 1 — action namespace + parsing | 1.5 wk | 2.5 wk |
| 2 — propagator builder | 1.5 wk | 4 wk |
| 3 — periodic BCs | 1 wk | 5 wk |
| 4 — IC machinery | 0.5 wk | 5.5 wk |
| 5 — Phase J extension to (t, x) | 2 wk | 7.5 wk |
| 6 — output API + tests + validation | 1 wk | 8.5 wk |

**Total v1: ~8.5 weeks of focused work.**

A useful intermediate milestone at Stage 4 end (5.5 wk in): the
framework can compute *tree-level* spatial correlators end-to-end
for any user-written theory, validated against free-theory closed
forms.  This is shippable as a v0.5 demo if you want to stop and
demo before doing the Phase J extension.

## Pitfalls to avoid

1. **Don't try to recognize every spatial operator in v1.**
   `Laplacian` only.  Higher-order operators (Cahn-Hilliard `∇⁴`,
   anisotropic, fractional) lose the closed-form `G(t, x)` heat
   kernel and require Stage 2's numerical-fallback path with
   careful UV/IR handling.  Defer.
2. **Don't mix infinite + periodic in the same theory.**  All
   spatial fields in a theory share one BC declaration in v1.
3. **Don't try to handle exclusion / hard-core constraints in
   v1.**  Lefèvre-Biroli §2.4 shows these need `n(1-n) e^{n̂_j -
   n̂_i}` vertices, which require action-parser extension beyond
   simple polynomial recognition.  Defer.
4. **Test the PBC → infinite limit early.**  This is the single
   best sanity check that BC + integration machinery are wired
   correctly.  If PBC at large `L` doesn't approach the infinite-
   domain answer, something's wrong with one or both.
5. **Stationary IC + Dirichlet BC may have subtleties.**  PBC is
   safe — the system is translation-invariant under both
   stationary IC and PBC.  Dirichlet BC plus stationary IC means
   the system relaxes to a non-uniform stationary profile.  v1
   supports PBC + infinite only, where stationary = spatially
   uniform.
6. **Keep dimension d as a parameter from day one** even though
   v1 ships with d=1 only.  Helps d=2 and d=3 extensions later.

## Reading list (before implementation starts)

| Reference | What it gives you |
|---|---|
| Andreanov 2006 §2 (continuum limit derivation) | Compact form of the gradient expansion that justifies treating `Laplacian` as a symbol in the user's action |
| Lefèvre-Biroli 2007 §2.2 (Dean's equation) | The cleanest worked example of how a phenomenological-looking spatial SDE emerges from the lattice MSR-JD construction |
| Lefèvre-Biroli 2007 §2.5 (initial conditions) | The `S_I` action term that future v1.5 transient-IC support will need |
| Lefèvre-Biroli 2007 §2.6 (boundary conditions) | The `S_B` action term for reservoir-coupled boundaries; out of v1 scope but useful framing |
| Lefèvre-Biroli 2007 §3.2 (Cole-Hopf operator level) | Why Daedalus can stay in `(ρ, ρ̂)` MSR-JD form throughout |

---

*Last updated: 2026-05-28.*

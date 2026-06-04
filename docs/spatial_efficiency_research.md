# Spatial loop-integral efficiency: a prioritized, cited roadmap

**Branch `spatial-extension`, June 2026.** A literature survey (adversarially
fact-checked, 25 sources, 19/25 claims confirmed) of methods to speed up the
spatial Schwinger-parametric chamber integral, mapped onto **our actual
backends** (`grid` / `mc` / `bessel` in `full_integrator.py`). Companion to
`docs/spatial_loop_integral_analytic_mc.md` (which has the Bessel-K
derivation) and `docs/spatial_pipeline.md` (backend matrix).

## The one finding that matters

> **Tropical Monte-Carlo sampling** (Borinsky 2020) is the *only* surveyed
> technique that **removes** our infinite-variance problem. Everything else
> (QMC, sparse grids, FFTLog, CSE/Horner, GPU) only **accelerates an integral
> that already converges** — stacking them on a still-infinite-variance
> integrand does nothing.

So the question "how do we make 2-loop **derivative-vertex** (KPZ / Model B /
Burgers) diagrams feasible" has a clear two-step answer: **(1) cure the
variance with tropical sampling; (2) then accelerate with QMC.** In that order.

## Our bottleneck, restated in our own terms

After the analytic reductions (Symanzik momentum → heat-kernel IFT → Bessel-K
radial), each diagram is an `(n_V + n_C)`-dimensional Schwinger-parametric
integral per causal chamber (`n_V` vertex times + `n_C` correlation-edge σ's):

- **`grid`** — dense product quadrature. `P = n_t^{n_V}·n_s^{n_C}`; a 2-loop
  KPZ diagram is 7-D, `P ≈ 1.8e8`/chamber → tens of GB → **OOM** (the memory
  guard now catches this instead of crashing).
- **`mc`** — importance-sampled `Exp(μ)` gaps + σ's. Plain `φⁿ`: works,
  memory-safe, <0.1%. **Derivative vertices: fails** — the Wick moment's
  `a=−M⁻¹N`, `Σ=(2DM)⁻¹` blow up as `det M → 0`, integrand `~U^{−5/2}` in a
  simplex corner → **infinite-variance / heavy-tailed** MC (46% bias @x=0).
- **`bessel`** — radial λ done analytically (Bessel-K, cures the
  *overall-scale* singularity), **angular simplex** done by Dirichlet MC.
  Handles derivative vertices at d=1, but the angular MC inherits the same
  `U(ŝ)→0` heavy tail on the simplex *boundary*.

**Where the surviving singularity lives (key insight for us).** Write the
Schwinger params as `(scale)·(direction) = λ·ŝ`. `U` is homogeneous, so
`U = λ^{deg}·U(ŝ)`. The `bessel` backend already integrates the **λ
(radial)** direction in closed form — so the `λ→0` part of `U→0` is *already
cured*. The residual `U^{−5/2}` blow-up is purely **angular**: it happens as
`ŝ` approaches a face where `U(ŝ)→0`. **Therefore the surgical target for
tropical sampling is the angular-simplex integral the `bessel` backend already
isolates** — a much smaller, more tractable object than a from-scratch
full-dimensional tropical integrator.

---

## Tier 1 — the variance CURE: Tropical Monte-Carlo sampling

**What it is.** Replace each Symanzik polynomial `p` (here `U` and `F`) by its
**tropical approximation** `p^tr`: set every coefficient to 1 and replace `+`
with `max` — e.g. `p = a·x₁²x₂ + b·x₁x₂x₃ + c·x₃³` ⟶
`p^tr = max(x₁²x₂, x₁x₂x₃, x₃³)`. Sample the normalized tropical measure
`μ^tr` (built from `U^tr`, `F^tr`), and form the **unbiased reweighted
estimator** `I = I^tr · E_{μ^tr}[R]`, `R = (true integrand)/(tropical
approx)`. [Borinsky 2020, Def. 6 / Alg. 2]

**Why it cures `det U→0` (the mechanism we need).** The tropical-approximation
theorem [feyntrop, Thm 3.1] gives a **two-sided bound** `C₁ ≤ |p(x)|/p^tr(x) ≤
C₂` on the simplex for any *completely non-vanishing* `p`. The first Symanzik
`U` is **always** completely non-vanishing on the positive simplex; `F` is too
in the Euclidean regime. Because the integrand carries exactly a `U^{−D/2}`
factor (`= U^{−5/2}` for our effective `D=5` derivative vertices), **that
singular factor is absorbed into the sampling measure** and the reweight ratio
`R` stays bounded as `U,F→0`. The sampling density automatically rises near the
integrable singularity. This converts our heavy-tailed/infinite-variance
integrand into a **finite-variance** one. [Borinsky 2020 Thm 8; feyntrop Thm
3.1; momtrop "integrable singularities … absorbed into the measure μ_G"]

**Why our pipeline is the *favorable* case.** The guarantees hold only in an
**effectively-Euclidean, real-positive-propagator** regime. Our MSR-JD
retarded heat-kernel propagators `G_R = θ(t)e^{−(μ+Dk²)t}` give **real,
positive Gaussian** momentum integrals and a **real Symanzik `F`** — so the
regime restriction is **satisfied by construction**, not an obstacle. (Contrast
genuine-Minkowski QFT, where the density is complex/non-smooth and tropical
sampling is hard.) [Borinsky 2020 intro + Remark 35; momtrop near eq. 2]

**Convergence / expected gain.** It is a *standard* `1/√N` finite-variance MC —
**not** asymptotically faster than plain MC, but with a drastically smaller
variance **constant**, and it **decouples cost from the chamber/sector count**.
Benchmark (momtrop, 3-loop "Mercedes", N=1e9): variance constant `C² = 0.519`
(tropical) vs `1.72e3` (multi-channel) vs `4.36e5` (naive) — **~10⁶× over
naive**. *Caveat:* that number is a momentum-space loop-tree-duality integrand,
**not** our exact chamber integral; the **mechanism** transfers (same `U,F`,
same `U^{−D/2}`), the **magnitude** will differ. [Borinsky 2020 Thm 5; momtrop
Mercedes table]

**Reference implementations to model on (don't reinvent):**
- **feyntrop** — C++ core (Borinsky) + Python (Munch), Feynman-parameter
  space. `arXiv:2302.08955`, CPC 292 (2023) 108874; `michaelborinsky.com/feyntrop`.
- **momtrop** — Rust, momentum space, the cleanest from-scratch template.
  `arXiv:2504.09613`, CPC 317 (2025) 109846; `github.com/alphal00p/momtrop`.
- Foundational theory: Borinsky, *Ann. Inst. Henri Poincaré D* 10 (2023)
  635-685, `arXiv:2008.12310`.

### ⚠️ Three open questions to resolve BEFORE implementing

The survey explicitly could **not** verify these for our integral — they are
the decisive implementation risks:

1. **Do the extra TIME dimensions fit?** Tropical machinery is stated for the
   pure-Symanzik (spatial-Schwinger) representation. Our integral has `n_V`
   **vertex-time** params on top of `n_C` σ's. Two routes: (a) treat the times
   as additional Schwinger-like params with their own monomial structure in an
   augmented Newton polytope, or (b) do the time/chamber integral separately
   (we partly do — causal chambers) and tropical-sample only the residual
   spatial-Schwinger simplex. **Route (b) aligns with our `bessel` backend's
   existing angular-simplex isolation** and is the lower-risk first target.
2. **Is the residual kernel square-integrable under `μ^tr`?** Taming `U,F` is
   *necessary but not sufficient*. After the heat-kernel IFT + Bessel-K + any
   derivative form factors, the leftover `f` must itself be square-integrable
   under `μ^tr`. Must be checked empirically for our KPZ/Model-B integrand
   (verify `f≡1` converges; watch for any *other* heavy tail).
3. **Tropical alone, or hybrid?** Our `U^{−5/2}` is an **integer-d integrable**
   singularity, not a dim-reg ε-pole. The two strongest "sector decomposition
   trivially restores finite variance" claims were **refuted** in verification
   precisely on this distinction. Tropical sampling very likely suffices, but a
   hybrid (monomialize `U` by one round of sector decomposition, *then* tropical
   / QMC the finite pieces) is the fallback if a tail survives.

---

## Tier 2 — ACCELERATORS (only after variance is finite)

These do **not** cure `det U→0`; apply them once Tier 1 has made the integrand
finite-variance.

- **Quasi-Monte-Carlo / lattice rules** *(strongest of the accelerators)*. The
  feyntrop authors themselves state swapping plain MC for QMC improves the
  runtime from `O(δ^{−2})` to `O(δ^{−1})` (error `~1/N` vs `~1/√N`). Stacks
  directly on tropical sampling. Concrete tooling: rank-1 shifted lattice rules
  (pySecDec's QMC; de Doncker & Yuasa), Sobol/Halton via `scipy.stats.qmc`,
  randomized-QMC for an error estimate. [feyntrop Outlook; confidence: *medium*
  — only the MC→QMC scaling statement was independently verified, not the
  specific lattice-rule refs.]
- **Fast Bessel-K / Hankel** *(targets our `bessel` radial factor)*. We already
  call `K_ν` per sample; for a basis of **shifted orders** `K_{P−m}` use the
  **recurrence** `K_{ν+1}=K_{ν−1}+(2ν/z)K_ν` (one `K_ν`,`K_{ν+1}` seed → whole
  family, avoids repeated special-function calls), and **DLMF §10.41** uniform
  asymptotics for large order/argument. FFTLog (Hamilton 2000; `mcfit`,
  `2-FAST`) is the log-spaced Hankel-transform tool if we ever batch a *family*
  of `|x|` outputs. [DLMF 10.41; FFTLog — confidence: *low*, not verified this
  pass, textbook-standard.]
- **Symbolic→numeric** *(targets the per-sample `U,F` + Wick-moment eval)*. We
  already use `lambdify(cse=True)`. Further: multivariate **Horner** form for
  the Symanzik polynomials (`multivar_horner`), and **numba**/`autowrap`
  LLVM-JIT for the hot kernel. Worthwhile only if the per-sample polynomial
  eval (not the sampling) is the measured hotspot. [confidence: *low*,
  question-framing.]
- **Sparse grids / adaptive cubature** as a `grid`-backend replacement: Smolyak
  (Gerstner-Griebel), Genz-Malik, Tasmanian. Big point-count cut for a *smooth*
  7-D integrand — **but they degrade exactly at our integrable boundary
  singularity**, so this helps the **plain-vertex** `grid` path, not the
  derivative-vertex corner. [confidence: *low*, question-framing.]
- **GPU / SIMD batching** (cupy/JAX) for the per-sample `U,F` + moment across
  millions of points — pure constant-factor throughput, last. [confidence:
  *low*.]

## Tier 3 — principled fallback: Sector decomposition

If tropical sampling alone leaves a residual tail (open question 3), **sector
decomposition** algebraically factorizes the `U→0` *corner* (endpoint)
singularity into a sum of finite, MC/QMC-integrable pieces — the canonical tool
for exactly "integrand blows up like `U^{−5/2}` in a corner." Use the
**geometric (Kaneko-Ueda)** variant (pySecDec default `geometric_ku`): it
recasts the problem into convex geometry and yields **fewer sectors** than
iterative Binoth-Heinrich. *Caveats:* heavier machinery; designed for dim-reg
ε-poles, so for our integer-d integrable singularity it's a valid but
slightly off-label use that must be validated empirically (the strong
"trivially restores finite variance" claims were refuted). Pair it with
tropical/QMC on the resulting finite pieces. [Heinrich `arXiv:0709.4092`;
pySecDec `arXiv:1712.05755`; Kaneko-Ueda CPC 181 (2010) 1352, `arXiv:0908.2897`.]

---

## Concrete next steps (mapped to our code)

1. **Prototype a `method='tropical'` angular backend** inside the `bessel`
   path: keep the analytic radial Bessel-K, but replace the **Dirichlet
   angular MC** (`_diagram_bessel_xs`) with a tropical proposal built from
   `U^tr(ŝ)` on the `(n−1)`-simplex. This is the smallest change that targets
   the actual surviving singularity, reuses our Symanzik machinery
   (`spatial_reduce.symanzik_polynomials` already gives `U`), and sidesteps
   open-question 1 (times already handled by chambers + Bessel-K). Validate
   variance on a 1-loop KPZ diagram (where grid is the oracle) before 2-loop.
2. **Resolve open-question 2 empirically**: with the prototype, check `f≡1`
   convergence and the tail of `R` for KPZ/Model-B.
3. **Only then** add **QMC** (Sobol via `scipy.stats.qmc`) under the tropical
   measure for the `O(δ^{−2})→O(δ^{−1})` win.
4. **Defer** sector decomposition, FFTLog, Horner/numba, sparse grids, GPU
   until 1-3 are measured — they are accelerators, not cures.

This is a research roadmap, not a tonight-implementation: the tropical backend
is a multi-step build with the three open questions to settle first, and the
choice to invest belongs to the user. Recorded here so the decision is grounded.

## Citations

| Ref | What | arXiv / DOI |
|-----|------|-------------|
| Borinsky 2020 | Tropical MC quadrature (foundational) | `arXiv:2008.12310`, AIHPD 10 (2023) 635 |
| feyntrop | Tropical Feynman integ., Minkowski | `arXiv:2302.08955`, CPC 292 (2023) 108874 |
| feyntrop (phys. region) | physical-region tropical | `arXiv:2310.19890` |
| momtrop | momentum-space tropical (Rust template) | `arXiv:2504.09613`, CPC 317 (2025) 109846 |
| Heinrich | sector decomposition review | `arXiv:0709.4092` |
| pySecDec | sector-decomp software | `arXiv:1712.05755` |
| Kaneko-Ueda | geometric sector decomposition | `arXiv:0908.2897`, CPC 181 (2010) 1352 |
| Hamilton | FFTLog | jila.colorado.edu/~ajsh/FFTLog |
| DLMF §10.41 | Bessel-K uniform asymptotics / recurrence | dlmf.nist.gov/10.41 |

*Full fact-check log (19 confirmed / 6 refuted claims, with votes) retained in
the session research artifact; refuted items were over-strong versions of true
claims — see Tier-1 open questions and the Tier-3 caveat, which already fold in
the surviving weaker forms.*

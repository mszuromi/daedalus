# Future Directions

A roadmap of possible v2 / later work for the MSR-JD Feynman-diagram
automation framework, organized by horizon and impact.  Each item lists
an effort estimate, the trigger that would motivate doing it, and the
specific code touch points (where they're known).

The tool's long-term goal is to handle **any stochastic field theory a
physicist would write down for a typical problem** — and to be credible
enough to claim that in a methods paper.  v1 (current state) covers the
common cases.  This document sketches what completing the rest looks
like.

---

## Current state (v1, May 2026)

What ships:

| Capability | Status |
|---|---|
| Gaussian white noise (independent, cross-correlated, multi-population) | ✓ |
| Non-Gaussian white noise via GTaS (κ³, κ⁴ via Bernoulli-Gaussian shifts) | ✓ |
| Gaussian colored noise — single Lorentzian `c·exp(-|τ|/τc)` | ✓ (Phase 1 Markovianization, May 2026) |
| Gaussian colored cross-correlated 2D Lorentzian | ✓ (Phase 1) |
| Linear-rational propagators (OU, OU+x³, OU+x⁵, 2D coupled, …) | ✓ |
| Hawkes-style `(exp(ñ) − 1)·φ(v)` actions | ✓ |
| Conv-vertex synaptic kernels | ✓ |
| Multi-population theories | ✓ |
| `k = 1` mean shifts, `k = 2` correlators, `k ≥ 3` higher cumulants | ✓ |
| Tree-level + 1-loop + 2-loop (white-noise theories) | ✓ |
| Tree-level + 1-loop (Phase-1-covered colored-noise theories) | ✓ |

What's tracked in the open-task list (`mcp__ccd_session__list_sessions` for
the current session, or `git log`'s referenced project memory notes):

* Pre-existing `spike_reset_k1_ell1` and `spike_reset_k2_ell0` regression
  fixture drift (documented; not user-visible at runtime).
* `μ = 1/τc` degenerate-pole edge case in the Markovianized propagator
  (workaround: perturb τc by 0.1 %; v2 fix below).
* Phase J ell≥1 hang on cross-correlated *colored* 2D OU before Phase 1
  shipped — resolved by Phase 1; no longer reachable in normal use.

---

## Near-term (v1.5 / pre-paper polish)  •  ~2–4 weeks total

These close the ergonomic and completeness gaps that matter most before
a wider release or a methods-paper submission.

### 1. Phase 1.5 — Multi-aux Markovian preprocessor  (~1 week)

Extend `pipeline/colored_to_markovian.py` from single-Lorentzian to:

* **Underdamped oscillatory** `c · exp(-|τ|/τc) · cos(ω₀τ)` — needs 2 aux
  fields with rotation matrix in the OU drift.  Physically: ringing
  membrane potentials, mechanical noise with a resonance, phonon-like
  fluctuations.
* **Sum-of-Lorentzians / double-exponential** `c₁·exp(-|τ|/t₁) +
  c₂·exp(-|τ|/t₂)` — needs 2 independent aux scalars.  Physically:
  multi-timescale fluctuations (fast electronic + slow ionic).
* **N-Lorentzian generalization** — straightforward bookkeeping
  extension; one aux field per pole-pair.

**Trigger**: a theory you actually need uses one of these kernels, OR
you're writing the paper claim "any Gaussian colored noise with rational
power spectrum".

**Effort**: ~150–200 LOC + tests, building on the v1 pattern matcher.

**Tracked as**: task #69 in current session.

### 2. Phase 2 — Direct kernel absorption in `poset_modesum`  (~1 week)

Per Agent 2's audit, extend `_extract_exp_mode` in
`msrjd/integration/time_domain/final_integral.py:203` to detect `abs(τ)`
and emit per-chamber mode pairs.  Loosen the `_analytic_eligible` gate
at line 3325–3334 to admit `noise_source_specs` when all kernels
successfully extract.  Pseudo-edge construction mirrors the existing
`ConvVertexType` synaptic-kernel code at lines 3486–3544.

**What this opens over Phase 1**:

* Polynomially-modulated kernels `τⁿ · exp(-|τ|/τc)` (Matérn-like) —
  these have no finite Markovian embedding.
* Other rational-in-ω kernels that don't admit clean Markovianization.
* The `μ = 1/τc` degenerate-pole bug is auto-fixed because
  `_exp_over_chain_simplex_polynomial` already has Taylor-fallback at
  `final_integral.py:1339-1486`.

**Trigger**: a methods-paper claim "any rational-in-ω noise spectrum
supported"; OR a theory needs Matérn-style polynomially-modulated
kernels; OR the degeneracy bug bites repeatedly.

**Effort**: ~150–250 LOC in one file.  Risk: medium — touches the
integrator's analytic-eligibility logic, which is well-tested.  Run the
full `tests/` suite after.

**Tracked as**: task #59 in current session (conditional defer).

### 3. Pre-filter zero-prefactor diagrams  (~1 day)

In `classify_coefficient_factors`, add an early
`SR(cp).subs(num_params).is_trivial_zero()` check that skips the
integration entirely.  At symmetric saddles many `(1, n_phys)` vertices
have coefficient `n · ε · xstar = 0` and the framework currently builds
their symbolic integrand, JIT-compiles it, and lets `scipy.nquad`
evaluate 0 across the polytope — pure waste.

**Trigger**: any theory with a symmetric saddle and ell ≥ 1.  This is
most theories.

**Effort**: ~10–30 LOC.  Expected 5–10× speedup at ell ≥ 1 for
zero-saddle theories.  Lowest risk on this list.

**Tracked as**: task #53.

### 4. Degenerate-pole handling for Markovianized propagator  (~3 days)

The `μ = 1/τc` exact case produces a multiplicity-2 pole in the
Markovianized propagator (Jordan block).  Current single-pole
infrastructure can't represent this; the result is numerical garbage
of magnitude `10¹⁵`.

**Two fix paths**:

1. **Detect and fail loudly** — add a squarefree check on `Q(ω)`; raise
   a clear error message instructing the user to perturb τc by 0.1 %.
   ~20 LOC; correct but ungraceful.
2. **Implement double-pole residue handling** — add multiplicity to the
   pole-list data structure, extend the residue computation to handle
   `1/(ω − p)²` terms via `f'(p)` instead of `f(p)/(...)`.  ~100–200
   LOC across `_propagator.py` and the analytic mode-sum integrators.

**Trigger**: the degeneracy keeps biting; OR Phase 2 ships (because the
chain-simplex polynomial fallback would auto-handle this anyway, making
the dedicated double-pole machinery less urgent).

**Tracked as**: task #68.

### 5. Spike-reset regression fixture re-freeze  (~1 day)

Two fixtures (`spike_reset_k1_ell1`, `spike_reset_k2_ell0`) in
`tests/test_phase_j_refactor_regression.py` carry stale frozen values
that no longer match the current integrator output.  Verified
historically as fixture drift, not code regression.  Re-derive the
correct frozen values from a known-good run and update the fixture
data.

**Trigger**: any time someone scans test output and worries about the
red.  Cheap to do.

---

## Medium-term (v2)  •  ~4–8 weeks

### 6. Path A — Non-Gaussian colored noise via filtered shot noise  (~2–3 weeks)

Most physically-interesting non-Gaussian colored processes are
filtered Poisson shot noise: a non-Gaussian white driver (random
amplitudes, Poisson arrival times) passed through a linear filter.
e.g. synaptic noise as a sum of exponentially-decaying impulses with
random amplitudes.

The framework already supports non-Gaussian *white* drivers via GTaS
shot noise (`coefficient` parameter encodes the higher cumulants).
Extending the Markovian preprocessor to detect Lorentzian kernels on
`order ≥ 3` CGF rows and generate an aux field driven by non-Gaussian
white noise should "just work" — the aux-field dynamics is still
linear, only the driver's statistics differ.

**Concrete deliverable**: `pipeline/colored_to_markovian.py` detects
patterns like

```
declare_cgf_term('shotnoise', response_legs=['xt','xt','xt'], order=3,
                 coefficient='c3', kernel='exp(-|τ₁|/τc) · exp(-|τ₂|/τc)')
```

and rewrites to

```
physical_field('ξ', ...)
declare_cgf_term('shotnoise_aux', response_legs=['ξt','ξt','ξt'], order=3,
                 coefficient='c3 · …', kernel='dirac_delta(τ₁)·dirac_delta(τ₂)')
equation(lhs='(Dt + 1/τc)*ξ', rhs='0')
# coupling -xt·ξ added to action
```

**Trigger**: a theory that needs colored shot noise (synaptic input,
compound-Poisson driver, level-crossing noise).  These are common in
computational neuroscience and statistical physics.

**Effort**: ~200 LOC for the preprocessor + ~200 LOC tests.  Builds
directly on the v1 / v1.5 infrastructure.

**Limitations**: requires the non-Gaussianity to be encodable as a
non-Gaussian white driver of a linear filter.  Doesn't cover truly
multiplicative noise, regime-switching, or non-stationary cases.

**Tracked as**: task #70.

### 7. Markovianize → Fourier conversion for spectral observables  (~2 weeks)

For applications where the natural deliverable is a power spectrum
$S(\omega) = \mathcal{F}\{\langle x(t)x(0)\rangle\}$ rather than a
time-domain correlator $\langle x(t)x(0)\rangle$, expose a Fourier
output path.  For Markovianized linear theories this is just reading
off the propagator residues — essentially free.

**Trigger**: a use case in fluctuation-dissipation studies, noise
spectroscopy, or any frequency-domain measurement comparison.

**Effort**: ~150 LOC.  No new math — just a different output formatting
of what the framework already computes.

### 8. Theory-library / gallery  (~1–2 weeks)

A curated set of `theories/gallery/*.theory.py` files representing the
canonical models in stochastic field theory, each with a matching
`notebooks/gallery_<name>.ipynb` that runs the theory end-to-end with
analytic + simulator validation.

Candidate canon (for a methods paper):

* 1D OU (white + colored, linear + cubic)
* 2D coupled OU (white + colored + cross-correlated)
* Cortical Poisson Hawkes (linear + quadratic φ)
* Spike-reset (white + colored)
* Bistable double-well Langevin
* Compound-Poisson-driven OU (non-Gaussian colored, via Path A)
* Underdamped Langevin (Phase 1.5 needed)

**Trigger**: writing a paper that needs a "table of theories the
framework handles".  Also the most-effective way to document the tool
for new users.

---

## Spatial extension — stochastic reaction-diffusion focus  •  4 weeks → multi-year, depending on depth

The framework currently handles fields ``φ(t)`` (functions of time
only); spatial structure is faked through populations (indexed
neurons / sites).  True spatial extension means fields ``φ(x, t)``
with continuous spatial coordinates — i.e. stochastic PDEs instead
of coupled stochastic ODEs.

**The primary goal here is reaction-diffusion (RD) equations**, of
the form

  ∂_t φ_i(x, t) = D_i ∇² φ_i + R_i({φ_j}) + η_i(x, t)

where ``D_i`` is the diffusion coefficient, ``R_i`` is a polynomial
reaction term, and ``η_i`` is Gaussian (white or colored) noise.
RD equations are the natural language for a huge class of
problems: stochastic Allen-Cahn / Cahn-Hilliard (pattern formation,
phase separation), Lotka-Volterra with spatial diffusion (ecology,
predator-prey), stochastic Turing patterns, neural field equations
(Wilson-Cowan with diffusion, neural-field cable equations),
calcium-wave dynamics in astrocytes, Belousov-Zhabotinsky chemistry,
cancer spreading models.

The MSR-JD action for stochastic RD has propagator

  G_i(ω, k) = 1 / (i ω + μ_i + D_i k²)

— **fully rational in both ω and k²**, parallel to the time-only OU
case.  Loop integrals become combined ``(ω, k)`` integrals: ω-part
handled by residue calculus (the existing time-only machinery
generalizes); k-part is a 1D radial integral after angular
integration, often closed-form for off-critical (gapped) theories.

**Two distinct RD frameworks worth flagging**:

1. **Phenomenological stochastic RD** — write the Langevin SDE
   directly with ``∇²`` and noise.  What most modeling papers do.
   Phase A' / B' below cover this.
2. **Doi-Peliti microscopic** — derive the field theory from a
   master equation for discrete particles.  Required for absorbing-
   state phase transitions, microscopic Poisson statistics,
   particle-number conservation.  Substantially different formalism
   (non-Hermitian action, creation / annihilation operators).  Not
   covered by simply extending the current framework; would be a
   parallel formalism.

### Phase A' — d=1 RD with Laplacian, off-critical  (~4–6 weeks)

First spatial milestone — restricted scope to keep the lift bounded
and shippable.  d=1 only (line); Gaussian noise (white or
Phase-1-covered colored); polynomial reaction terms (already
supported by the action parser); positive mass ``μ > 0`` so all
k-integrals converge naturally (no UV regularization needed yet).

**Architecture additions**:

* ``physical_field('phi', spatial_dim=1, kinetic_operator='laplacian')``
  declares a continuous spatial field with diffusion as kinetic term.
* A new ``diffusion`` parameter type alongside the scalar /
  vector / matrix ones.
* Propagator builder threads ``k`` alongside ``ω`` symbolically.
  ``K_ft(ω, k) = K_ft_temporal(ω) + D·k²`` entry-wise.
* Phase J's residue closure over ``ω`` is unchanged; an additional
  1D radial k-integral runs over each loop momentum.  Many cases
  closed-form; Sage's ``integrate`` handles the standard ones
  directly.

**Test theory** — stochastic Allen-Cahn::

  ∂_t φ = D ∂²_x φ + μ φ − λ φ³ + η,    ⟨η η⟩ = 2T δ(x−x') δ(t−t')

with closed-form benchmarks for ⟨φ(x, t) φ(0, 0)⟩ at tree and
1-loop via residues + radial k-integral.

**Code touch points**:

* ``pipeline/_propagator.py`` — k-space symbolic threading (~200 LOC)
* ``pipeline/theory.py`` — new keyword args on ``physical_field``
  and ``parameter``; new ``kinetic_operator`` field on the spec
* ``msrjd/integration/time_domain/final_integral.py`` (or a new
  ``frequency_momentum_integral.py``) — k-integral evaluator
  alongside the existing ω-residue path (~300 LOC)
* ``msrjd/core/field_theory.py`` — recognize spatial fields in the
  action expansion so ``∂²_x`` becomes ``-k²`` in Fourier
* New tests for free-theory analytic checks + 1-loop Allen-Cahn

**Trigger**: advisor wants spatial RD demos; a specific 1D RD
problem (calcium dynamics on a dendrite, 1D neural field, 1D
Lotka-Volterra) the user wants to compute on.

### Phase B' — d=2,3 RD off-critical  (~6–8 weeks)

Generalize the k-integrals to d-dimensional radial: ``∫ dᵈk →
S_{d-1} ∫₀^∞ dk · k^{d-1}`` after angular integration.  Most loops
still have closed forms or reduce to dilogarithms/Γ-functions for
mass-regulated theories.

**Test theory** — Cahn-Hilliard above critical, or 2D stochastic
Lotka-Volterra (predator-prey on a plane) at the Langevin level.

**New machinery**:

* d-dimensional radial integration utilities (1-loop = standard
  textbook; 2-loop = Feynman-parameter machinery)
* Convergence diagnostics — for marginally finite integrals,
  detect when a UV cutoff becomes necessary

**Trigger**: a specific 2D / 3D RD problem the user / advisor wants
(spatial cortex models, 2D pattern formation, 3D reaction-diffusion
in a tissue volume).

### Phase C' — Critical RD / pattern formation / Doi-Peliti  (~3–6 months)

Three subsequent tracks once Phase A'/B' are in place:

1. **Critical RD** — Turing instability, ε-expansion, anomalous
   dimensions.  RG flow machinery (overlap with ``FeynCalc``,
   ``FORM``, ``QGRAF`` in HEP).
2. **Doi-Peliti microscopic formalism** — non-Hermitian action,
   creation / annihilation operator algebra.  Required for
   absorbing-state phase transitions (directed percolation,
   contact process, branching annihilating random walks).  This
   is a parallel formalism that would either coexist with the
   MSR-JD machinery or be a separate ``daedalus.doi_peliti``
   submodule.
3. **Non-equilibrium critical exponents** — extracted from RG
   flows of stochastic RD theories.  KPZ class, directed
   percolation, model A/B/C/D/E from Hohenberg-Halperin.

This is true research-tool territory.

**Trigger**: a methods paper that claims competitive coverage with
hand-derived RG results in critical RD or non-equilibrium statistical
mechanics.

### RD applications that motivate this work

**Neuroscience**:

* **Neural field equations** (Wilson-Cowan with diffusion,
  cable-equation models of dendritic propagation) — Phase A' (1D
  dendrite) or Phase B' (cortical sheet in 2D).
* **Calcium dynamics on dendrites and in astrocytic networks** —
  1D / 2D RD with cubic Ca²⁺ release.  Phase A'/B'.
* **Traveling wave dispersion in cortex** — continuum dispersion
  relation ω(k); Phase B'.
* **Spreading depression** — RD with bistable kinetics;
  Phase A' (1D) or B' (2D).
* **Critical brain dynamics (avalanches, scaling)** — Phase C'.

**Outside neuroscience**:

* **Stochastic Allen-Cahn / Cahn-Hilliard** (pattern formation,
  phase separation) — Phase A'/B'; canonical RD test problems.
* **Stochastic Lotka-Volterra** (predator-prey, ecology) — Phase
  A'/B'.
* **Stochastic Turing patterns** — Phase B' (above threshold)
  or C' (at threshold).
* **Belousov-Zhabotinsky / chemical reaction-diffusion** — Phase B'.
* **Population dynamics with diffusion** — Phase A'/B' (continuous
  limit of birth-death processes).
* **Tumor / pathogen spreading models** — Phase A'/B'.

### Recommended starting point

For an advisor-facing demo of "Daedalus does reaction-diffusion",
commit to **Phase A' end-to-end** on a stochastic Allen-Cahn 1D
theory.  ~4-6 weeks of focused work.  Deliverables:

1. The framework accepts a theory file with
   ``physical_field('phi', spatial_dim=1, kinetic_operator='laplacian')``
2. Compute the equal-time 2-point correlator
   ``⟨φ(x, t) φ(0, t)⟩`` and the spatio-temporal correlator
   ``⟨φ(x, t) φ(0, 0)⟩`` at tree and 1-loop.
3. Match against the closed-form free-theory result + known 1-loop
   λφ⁴ corrections.
4. Sweep ``μ`` toward 0 and demonstrate the framework continues to
   produce sensible results above (but not at) the critical point.

That demo establishes "we do reaction-diffusion".  Phase B'
(2D, 3D) and Phase C' (critical / Doi-Peliti) wait for specific
motivating physics problems.

### What the current framework gets you for free, RD-wise

Two zero-work demos you can show your advisor today:

1. **Pseudo-1D RD on a small lattice via multi-population**.
   Declare a population ``L`` of size N = 10, declare a coupling
   matrix ``w[i,j]`` that's a discretized Laplacian
   (``-(2 δ_{i,j} − δ_{i,j+1} − δ_{i,j-1})``), declare a cubic
   reaction term ``-eps · x[i]^3`` in the action.  This is a
   working stochastic Allen-Cahn on a 10-site lattice — runs in
   seconds at 1-loop with the current framework.  Not the
   continuum limit but real RD-flavor physics.
2. **Lattice + colored noise** is already supported via Phase 1
   Markovianization.  Drives the demo above with OU-colored noise
   on each site without any new code.

These are the natural warmups to motivate Phase A'.

---

## Long-term (research-scale)  •  multi-month or multi-year

These are real-research-program items.  Each opens a class of theories
that no current framework handles cleanly.

### 9. Path B — Direct κⁿ smooth-kernel integrator extension  (~2–3 months)

Extend the mode-sum integrators to handle n-time smooth kernels with
retardation constraints.  Currently the framework's analytic
integrators (`interval_modesum`, `polygon_modesum`, `poset_modesum`)
handle κ² (= 2-leg) with smooth kernels via the Phase 2 chamber-split
trick.  For κⁿ (n ≥ 3) smooth, you need:

* n-leg noise vertex with (n−1) τ_v integration variables
* Per-leg routing analogous to Phase 1 but with n−1 internal times
* New analytic closed forms for the n-leg case (m=3 chain-simplex
  extends; m=4+ requires real work)

**What this opens**: non-Gaussian colored noise via direct integration
(no Markovianization needed) — closes the "any colored noise process"
claim cleanly.

**Trigger**: a specific theory needs a κⁿ shape that isn't
representable as a filtered-shot-noise process.  Or a reviewer asks
"why don't you handle this case directly?".

### 10. Path C — Smarter numerical fallback  (~1 month)

For kernels that fall outside the analytic-mode-sum scope, replace
`scipy.nquad` with something better:

* **Vectorized Monte Carlo** with importance sampling on the kernel
  modes (good for high-dimensional polytopes).
* **Tensor-network contraction** of the multi-time integrand (good for
  small-but-non-rational integrands).
* **Cubature** instead of Gauss-Kronrod (handles smooth-then-sharp
  transitions better than adaptive subdivision).

**Trigger**: hard-to-handle kernels keep coming up.  Less elegant than
extending the analytic path but more general.

### 11. Lévy / heavy-tailed white noise  (~3 months, research-level)

Stable-distributed white noise with diverging higher cumulants.
Requires:

* New CGF representation (characteristic function instead of cumulant
  expansion — κⁿ doesn't exist for n ≥ 2 for stable-α with α < 2)
* Generalized diagram rules
* Analytic computation of fractional-moment correlators

**Trigger**: a use case in anomalous diffusion, finance, or
extreme-value physics.

### 12. Non-stationary noise  (~3–6 months, architectural)

Currently the framework assumes stationarity: kernels depend on `τ = t
− t'`, not on `t` and `t'` separately.  Non-stationary noise (kernels
explicitly time-dependent, e.g. after a quench or stimulus onset)
breaks this assumption fundamentally.

**Requires**:

* New time-coordinate convention throughout the integrator
* Mode-sum integrators generalized to absolute-time kernels
* MF saddle becomes time-dependent (initial-value problem instead of
  steady-state DAE)

**Trigger**: extreme — this is a major architectural pivot.  Worth
discussing if a paper specifically targets transient / driven systems.

### 13. Fractional Brownian motion / long-memory noise  (~3 months, research-level)

Kernels with power-law decay `1/|τ|^α`.  Power spectrum
`1/|ω|^(1−α)` — non-rational, not Markovianizable.  Requires either:

* Fractional calculus formulation of the field theory
* Or approximation by sum-of-Lorentzians with a wide spread of τc
  values (controlled approximation; not exact)

**Trigger**: anomalous diffusion problems, 1/f noise studies, glassy
dynamics.

### 14. Full Fourier-domain Phase J rewrite  (~2-3 months)

Per Agent 3's audit, a parallel Phase J integrator working entirely in
frequency space via residue calculus.  For rational-kernel theories
this is essentially equivalent to the current time-domain mode-sum
but expressed differently.  Substantial reorganization with limited
new capability — Agent 3 recommended deprioritizing.

**Trigger**: pure-spectral observables become the dominant use case
(susceptibilities, response functions, fluctuation-dissipation
applications).  Unlikely.

---

## Cross-cutting

### A. UI polish

* **Open-in-runner-notebook button** after Save: writes
  `.theories/.last_built` and opens the runner notebook.  Sketched in
  the May 2026 UI rewrite review; not yet implemented.
* **Structured CGF input** that mirrors the structured Defaults tab —
  separate widgets for amplitude, kernel shape, parameter mapping
  instead of free-form text.  Lower opacity for new users.
* **Per-field validation** with live syntax-checking of the action and
  MF equation textareas (the `expression_input` widget at
  `pipeline/ui/widgets.py:73-115` is built for this but not wired into
  the multi-line textareas yet).
* **Save-as / version-control of theory files** — currently `.theory.py`
  files are flat with no versioning metadata.  A header annotation
  with framework-version + write-timestamp would help reproducibility.

### B. Performance

* **Stage 4 subset-signature caching** (mentioned in
  `pipeline/_propagator.py` comments; not implemented).  Reuses
  compiled fast-callable closures across diagrams with isomorphic
  subset structure.
* **Vectorized analytic paths** — the m=3 chain-simplex inner loop is
  written for clarity; vectorizing the index loops with NumPy would
  ~3-5× the per-diagram speed at ell ≥ 1.
* **Parallel-pool work-stealing** — current pool uses fork + map, which
  blocks on the slowest worker.  Work-stealing would smooth out
  variance across diagrams of widely-different complexity.

### C. Documentation

* **Tutorial series**: "Writing your first theory" → "Adding colored
  noise" → "Cross-correlated 2D" → "Custom non-linearities".  Builds
  on the in-notebook documentation that was added to
  `notebooks/theory_builder.ipynb` in May 2026.
* **Theory-file format spec** — currently the `.theory.py` schema is
  documented implicitly via examples.  A formal reference (every
  builder method, every metadata key, expected ranges) would help.
* **Methods paper** — the canonical writeup of the framework's
  capabilities, limitations, and validation suite.  Drives the
  prioritization decisions above (which "marketing claims" need to be
  defensible).

---

## Recommended sequencing

If your goal is **a methods-paper submission in ~6 months**, my
recommendation:

| Month | Work |
|---|---|
| 1 | Phase 1.5 (underdamped + double-exp + sum-of-Lorentzians) + Phase 2 (poset_modesum kernel absorption) — closes "any rational-in-ω colored kernel" |
| 2 | Path A (non-Gaussian colored via filtered shot noise) — closes "non-Gaussian colored noise" |
| 3 | Theory-library / gallery — produces the table of validated theories for the paper |
| 4 | Zero-prefactor pre-filter + degenerate-pole handling + fixture re-freeze — polish |
| 5 | Tutorial documentation + methods-paper draft |
| 6 | Submission |

If your goal is **single-user research workflow**, keep the v1.5 items
on the back burner until a specific theory demands them.  Phase 1
already covers everything you've worked on so far.

---

*Last updated: 2026-05-27.  See `docs/CHANGELOG.md` for the
implementation history.*

# Backend C — engineering design (the general loop evaluator)

**What this is.** The implementation plan for backend **C** — the
Schwinger/parametric loop integrator that lifts the spatial pipeline toward
**arbitrary loop order `L` and spatial dimension `d`**.

**Core vision (the scope decision — `backend_C_math.md` §0).** The core engine is
**automated perturbative MSR-JD for finite-scale SPDEs with physically meaningful
cutoffs** (neural fields, lattice/RDME simulators, finite synaptic/axonal ranges,
colored noise). In that setting the UV cutoff is *physical*, loop integrals are
simply **finite**, and **no renormalization is needed** — the core is: Symanzik
momentum reduction (closed-form for a smooth Gaussian cutoff; numerical for
hard/lattice cutoffs, but finite either way — §4 C1) → finite causal-time
parameter integral → adaptive numerics. **Renormalization (sector decomposition + dim-reg + RG) is an
optional module for continuum-critical theories (Regime 3), off the critical
path.** This is the major simplification: the hardest, research-grade piece is no
longer required for the core.

**The core deliverable, stated precisely:** *a topology-generic, finite-cutoff,
time-domain Symanzik evaluator for diffusive MSR-JD diagrams at tree / 1-loop and
selected higher-loop order, validated against simulations at the same cutoff.*
(Not "arbitrary loop order" as a deliverable — see §6, III.3.)

This is the design that `docs/spatial_v2_architecture.md` §5 (option C) / D2 names
as the long-term target. Math foundation: `docs/backend_C_math.md`.

**Status.** Design. No C code exists yet; the 1-loop momentum core
(`loop_parametric.gaussian_momentum_integral`) and the topology/routing
(`momentum_routing.route_momenta`) it generalizes are built and validated.

---

## 1. Why C — the limitation it removes

Today the spatial pipeline is `k=2`, `ℓ≤1`, `d=1`. The 1-loop machinery
(`loop_parametric` / `loop_dyson`) is **single-loop and bespoke per topology**.
The three temporal backends scale differently (architecture §5):

- **A** (Phase-J ordering polytope, per-chamber quadrature): general but the
  per-chamber quadrature dimension grows with vertices → expensive at high `L`.
- **B** (`loop_dyson` explicit Dyson convolutions): fastest where it applies, but
  **bespoke per topology** — every new self-energy is hand-coded.
- **C** (this doc): the momentum integral is *always* Gaussian (Symanzik), so it
  scales to high `L`; `d` is a parameter; **a physical cutoff makes the residual
  integral finite** (Regimes 1–2 — no renormalization), with sector decomposition
  reserved for the optional continuum-critical case. The only backend that makes
  the pipeline more than a 1-loop, `d=1` toy.

The four-axis extension study found that **`ℓ>1` and `d>1` both converge on C**,
and that C also (i) turns the `d>1` UV "cutoff trap" into a *feature* — the
physical cutoff is simply respected and the loop is finite (the simulator's own
resolution sets it), and (ii) **avoids the close-pair bug at its root by never
forming pole-difference `1/(λᵢ−λⱼ)` denominators during loop integration**
(rather than patching it per-diagram — see `backend_C_math.md` §4b: close-pair is
a representation artifact, not a boundary divergence). So C is the spine; the
other extensions hang off it.

---

## 2. Scope — what C does and does not do

**Does:** the *loop evaluation*. Given a typed diagram + its momentum routing,
produce the self-energy / correlator contribution — the **momentum reduction
exact at arbitrary `(L, d)`**, then a **finite** causal-time parameter integral
(physical cutoff, Regimes 1–2) by adaptive numerics. Subsumes: the `ℓ>1` axis,
the `d>1` axis (the cutoff is respected, not removed), and **structural
avoidance** of close-pair (no pole-difference denominators are formed).
*Optional add-on:* the Regime-3 renormalization module (sector decomposition +
dim-reg) for continuum-critical theories.

**Does NOT** (separate workstreams that *compose* with C):
- `k>2` **output transform** — the external multi-momentum → multi-position
  Fourier transform is a different integral (the *external* transform, not the
  loop). Tracked under the `k>2` axis.
- **non-Gaussian noise vertex content** — authoring + enumeration of `φ̃ⁿ`
  sources; already generic for local cumulants (see the noise-axis assessment).

---

## 3. Ordering & dependencies (does this wait on Phases I/II?)

**No strict sequence.** The dependency logic:

- **Do first (feeds C):** the thin Phase-I slice — *retire the close-pair
  sidecar* (replace the `compute_correction_td`-based coupling extraction in
  `pipeline_bridge.compute_spatial_correlator_bubble` with an analytic `M(Γ)`
  read) and the *Dyson resummation* of the existing bubble. C reuses both
  (normalization conventions; the resummation structure). Low risk, high leverage.
- **Do NOT do first — fold into C:** the bespoke numerical `d=2` angular
  quadrature (old "Phase II"). C delivers `d=2/3` through the Symanzik `U^{−d/2}`
  exponent far more cleanly; building the angular-quadrature hack first is
  throwaway work.
- **Independent side-tracks (anytime):** `k>2` momentum-space spine; local
  non-Gaussian noise lock; Path-A colored-shot-noise Markovianization.

So the route is: **Phase-I slice → C**, with the orthogonal axes interleaved as
convenient.

---

## 4. Component architecture (C0–C4)

Slots into the `spatial_reduce.py` / `temporal_integrate.py` modules the
architecture §8 already reserves, plus one new module for sector decomposition.
`temporal_integrate` is a **pluggable strategy `A|B|C`** (the D2 reversal seam):
C is added *alongside* A/B; B stays the 1-loop fast-path **and the oracle** (C
must match B on the bubble to ~1e-6).

| Stage | Module | Role | Risk |
|---|---|---|---|
| **C0** Graph → Symanzik | `spatial_reduce.py` | From `route_momenta` edge forms, build `M,N,Q(w)` and `U=det M`, `F`. Generalizes the scalar `U=Σa²w` to the matrix case. | LOW |
| **C1** Momentum integral | `spatial_reduce.py` | Closed `(4πD)^{−Ld/2} U^{−d/2} e^{−DF/U} e^{−μΣw}` for the **smooth-Gaussian / continuum** case (`d`-general, any `L`; promotes `gaussian_momentum_integral` to `det/inverse`). For a **hard** cutoff or **lattice** dispersion the loop stays finite but C1 falls back to incomplete-Gaussian / numerical Brillouin-zone evaluation (not the pure `U,F` form). | LOW–MED |
| **C2** Causal time-simplex | `temporal_integrate.py` | Assemble the residual `∫∏dw` with retarded `θ`-orderings (reuse Phase-J chamber enumeration) + correlation-edge Schwinger limits + external `τ`, **with the physical cutoff applied** (Gaussian regulator → weight shift; hard/lattice → finite domain). **The MSR-JD-specific part.** | MED |
| **C3-lite** Finite-cutoff quadrature **(CORE)** | `temporal_integrate.py` | Robust adaptive quadrature on the *finite* causal parameter integral (the cutoff already removed the singularity). No `ε`-poles, no subtraction. This is the core Regime-1/2 evaluator. | MED |
| **C3-full** Sector decomposition + dim-reg **(OPTIONAL — Regime 3)** | `sector_decomp.py` (new) | *Only for continuum-critical theories.* Factorize the UV endpoint/sub-divergences (`U→0`; forest formula), extract `ε`-poles (`d=d_c−2ε`), renormalize. The 1-loop UV sliver is the prototype. **Off the core critical path.** | HIGH (research) |
| **C4** Numerical eval (+ optional renorm) | `temporal_integrate.py` (+ `renorm.py`) | QMC/adaptive evaluation; for C3-full, assemble the `ε`-Laurent series and absorb poles into `Z`-factors. | LOW–MED (core); MED (renorm) |

**Risk profile in one line: the CORE (C0/C1/C2/C3-lite/C4) is engineering — linear
algebra (the exact momentum reduction) + a finite causal-time quadrature; the only
research-grade piece (C3-full: real-time sector decomposition) is OPTIONAL and
serves continuum-critical theories alone.** Two consequences: (1) with a physical
cutoff the loop is finite by construction (a Gaussian regulator even keeps the
momentum integral closed-form and the weights bounded away from 0 — no singularity
arises), so the core never touches `ε`-expansions; (2) the **close-pair** pathology
is avoided in **C2 — *provided* products of kernels are kept in parametric form
and NOT partial-fractioned into modal pole differences**; in particular the
multi-pole colored-noise correlation edge (§7) must be carried parametrically (or
summed by stable routines), since implementing it by naive residue differences
re-introduces the bug. Any step that does form `1/(λᵢ−λⱼ)` gets stable
divided-difference / confluent evaluation, never a divide-by-`(m−m')`.

---

## 5. Build vs borrow

- **C0/C1 (Symanzik):** in-house. Our graphs are small and our "masses" are just
  `μ+Dk²` with exponential propagators — the polynomial-from-graph step is short;
  borrow only the *formulas* (Bogner–Weinzierl).
- **C2 (causal time-simplex):** in-house. No external tool models retarded
  `θ`-orderings; this is our physics, and the 1-loop `loop_dyson` time route is
  the template.
- **C3-lite (finite-cutoff quadrature — CORE):** in-house, `scipy`/QMC adaptive
  quadrature on the finite causal parameter integral. No external dependency; this
  is the deliverable for Regimes 1–2.
- **C3-full (sector decomposition — OPTIONAL, Regime 3):** only if/when continuum
  critical exponents are wanted. The algorithm is intricate: **(a)** drive
  **pySecDec** on the parametric integrand, or **(b)** focused in-house for the
  causal structure. Deferred; not on the core path. If pursued, prototype with (a)
  and decide (b) only if pySecDec's Euclidean assumptions fight the causal
  structure.

---

## 6. Milestone / validation ladder

Every milestone is a *checkable* result against an independent oracle, so the
research never runs blind:

- **III.0 — reproduce the 1-loop bubble** (reaction-diffusion `φ̃φ²`, `d=1`)
  through C0→C1→C2→C4. *Oracle:* backend B (`loop_dyson`, B≈0.99) + the 1D
  simulator. Proves the Symanzik pipeline on known-good; **no new physics, all
  new plumbing.**
- **III.1 — 2-loop sunset**, reaction-diffusion, `d=1`, equal-time `δC(q,0)`.
  *Oracle:* a brute-force `∫dℓ₁dℓ₂` reference (slow but correct) at a few `q` +
  the simulator at a 2-loop-visible coupling. First genuinely new result; first
  real test of C2 at `L=2`.
- **III.2 — turn on `d=2/3` with a finite cutoff** (the `U^{−d/2}` exponent at the
  simulator's `k_max`, or a Gaussian connectivity kernel). *Oracle:* a `d=2/d=3`
  simulator at the **matched cutoff** + the static closed forms (`K₀` in `d=2`,
  Yukawa in `d=3`) for the tree part. **No `ε`-expansion needed** — finite numbers
  vs finite numbers. This is the core `d>1` deliverable.
- **III.3 — topology-generic finite-cutoff evaluator for small-to-moderate `L`,
  arbitrary `d` as a parameter**, driven straight from `enumerate_unique_diagrams`
  + `route_momenta`. (Arbitrary loop order is *conceptually* true for C0/C1 but is
  **not** a realistic deliverable — the causal-time quadrature dimension and
  diagram count grow with `L`; target tree / 1-loop and *selected* higher-loop
  topologies, not "all `L`.") This is the core engine done.
- **III.R (optional, Regime 3) — continuum renormalization.** Only if critical
  exponents matter: add C3-full (sector decomposition + dim-reg), validate the
  leading `ε`-poles against the known critical-dynamics RG of Model A/B (Täuber).
  Separate, later, off the core path; **build-vs-borrow (pySecDec) decided here.**

---

## 7. Interfaces / data contracts

Aligns with the §4 integrator interface (the D1/D2 reversal seam):

```
spatial_reduce.reduce(routing: RoutingResult,
                      edge_kinds: dict[edge, {'retarded'|'correlation'}],
                      d: int) -> SymanzikForm
    # SymanzikForm: U(w), F(w,q), prefactor(L,d), edge→duration-parameter semantics
    #   retarded    → fixed Δt
    #   correlation → integrated s ≥ |Δt|; for MULTI-POLE / Markovian-embedded
    #     colored noise (and multi-field), a correlation edge is a FINITE SUM over
    #     modes, each with its own (s_e, residue, mass m_{a,k}) — the reduction
    #     applies term-by-term (do NOT hard-code the single-pole OU form).

temporal_integrate.integrate(symanzik: SymanzikForm,
                             orderings,            # Phase-J chambers (causal θ's)
                             ext_times, num_params,
                             cutoff: Cutoff,       # see the explicit contract below
                             renormalize=False) -> result
    # CORE (renormalize=False, Regimes 1–2): a FINITE value/array — the cutoff
    #   makes the parameter integral finite; just quadrature.
    # OPTIONAL (renormalize=True, Regime 3): an EpsLaurent {−p:…, 0: finite}
    #   (poles + renormalized finite part) via C3-full.
```

**The cutoff is a first-class input, because each mode changes the evaluator
differently** (closed-form vs numerical, §4 C1):

```python
cutoff = {
    "type":       "gaussian_edge" | "hard_spherical" | "lattice_bz",
    "sigma":      float,   # gaussian_edge: per-edge e^{−σ²k²} → w_e += σ²/D (closed form kept)
    "kmax":       float,   # hard_spherical: |k|<kmax  → incomplete-Gaussian / numerical radial
    "a":          float,   # lattice_bz: mesh spacing → ∫ over Brillouin zone [−π/a, π/a]^d
    "dispersion": callable # lattice_bz: m_k = μ + (2D/a²)Σ_i(1−cos k_i a) (or a custom Ĵ(k))
}
# 'gaussian_edge' is the friendliest (stays closed-form); the other two stay
# finite but force the C1 numerical fallback. To validate against a grid
# simulator, use type='lattice_bz' with the simulator's a (or kmax = πN/L).
```

Inputs already exist: `route_momenta` returns `edge_momenta` (linear in
`q_syms`, `loop_syms`) and `n_loops`; `edge_k2()` gives `k_e²`. The
edge-kind tagging comes from the diagram typing (response vs correlation lines).

---

## 8. Risks & open questions (honest)

- **Scope first (this de-risks everything below).** The CORE engine (Regimes
  1–2, finite cutoff) needs no renormalization — C2 + C3-lite are a finite causal
  quadrature, engineering not research. The *only* research-grade open piece is
  **C3-full: real-time / causal Symanzik + sector decomposition** for the
  continuum-critical limit (Regime 3). The Euclidean polynomials are textbook;
  combining them with retarded `θ`-orderings + Keldysh structure is *not* a
  packaged result (architecture §9) — but it is **optional and off the core path**,
  so it no longer gates the project.
- **Time-domain vs frequency-domain.** Two ways to set up C2: keep time explicit
  (causal simplex, as above — *recommended*, matches the validated 1-loop), or
  Fourier to `ω` and do `∫dω` with retarded poles by contour. Time-domain avoids
  the contour bookkeeping and inherits the close-pair-free property.
- **Renormalization scheme (Regime 3 / III.R only).** If C3-full is ever pursued,
  pin conventions (minimal subtraction; the dynamic `Z_φ,Z_D,Z_μ,Z_g,Z_T`) against
  a reference Model-A/B computation before trusting finite parts. Irrelevant to the
  core.
- **Performance.** Core cost is the causal-time quadrature dimension (`#vertices`
  + `#correlation-edges`) at high `L`; mitigate with adaptive/QMC. (Sector-count
  growth is a Regime-3/III.R concern only.)
- **Cutoff is a first-class input, not a fudge.** Every core result is reported
  *at* its cutoff and validated against a simulator at the **same** cutoff — so the
  oracle (the simulator) stays live all the way through III.3, which is the chief
  practical advantage of the finite-scale scope. (III.R alone leans on known
  `ε`-coefficients / Ward identities instead.)

---

## 9. References

Same as `docs/spatial_v2_architecture.md` §9: Bogner–Weinzierl
[1002.3458](https://arxiv.org/abs/1002.3458) (Symanzik `U,F`); Weinzierl
[2201.03593](https://arxiv.org/abs/2201.03593) (representations); pySecDec
[2202.13647](https://arxiv.org/abs/2202.13647) (sector decomposition + QMC);
Täuber *Critical Dynamics* + [cond-mat/0511743](https://arxiv.org/abs/cond-mat/0511743)
(MSR-JD dynamic renormalization); Kamenev (Keldysh structure).

# Backend C — engineering design (the general loop evaluator)

**What this is.** The implementation plan for backend **C** — the
Schwinger/parametric loop integrator that lifts the spatial pipeline from
"`ℓ≤1`, `d=1`" to **arbitrary loop order `L` and spatial dimension `d`, with
systematic UV renormalization.** This is the design that
`docs/spatial_v2_architecture.md` §5 (option C) / D2 names as the long-term
target. Math foundation: `docs/backend_C_math.md`.

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
  scales to high `L`; `d` is a parameter; UV divergences are handled by sector
  decomposition + dim-reg → **renormalized** results. The only backend that makes
  the pipeline more than a 1-loop, `d=1` toy.

The four-axis extension study found that **`ℓ>1` and `d>1` both converge on C**,
and that C also (i) systematizes the UV audit that `d>1` otherwise leaves as a
silent cutoff trap, and (ii) cures the close-pair bug at its root rather than
per-diagram. So C is the spine; the other extensions hang off it.

---

## 2. Scope — what C does and does not do

**Does:** the *loop evaluation*. Given a typed diagram + its momentum routing,
produce the renormalized self-energy / correlator contribution at arbitrary
`(L, d)`. Subsumes: the `ℓ>1` axis, the `d>1` axis, the UV-renormalization audit,
and a principled close-pair cure.

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
| **C1** Momentum integral | `spatial_reduce.py` | `(4πD)^{−Ld/2} U^{−d/2} e^{−DF/U} e^{−μΣw}`, `d`-general, any `L`. Promotes `gaussian_momentum_integral` to `det/inverse`. | LOW–MED |
| **C2** Causal time-simplex | `temporal_integrate.py` | Assemble the residual `∫∏dw` with retarded `θ`-orderings (reuse Phase-J chamber enumeration) + correlation-edge Schwinger limits + external `τ`. **The MSR-JD-specific part.** | MED–HIGH |
| **C3** Sector decomposition | `sector_decomp.py` (new) | Factorize `U→0` (UV) and near-degenerate (close-pair) singularities; remap to unit cube; extract `ε`-poles (`d=d_c−2ε`); finite integrands. The 1-loop sliver is the prototype. | **HIGH** (research core) |
| **C4** Numerical eval + renorm | `temporal_integrate.py` (+ `renorm.py`) | QMC/adaptive on finite sectors; assemble `ε`-Laurent series; absorb poles into `Z`-factors (minimal subtraction); return renormalized correlator. | MED |

`d=1` at `L=1` needs **no** sector decomposition (the integrable singularity is
handled by the existing sliver) — so C0→C1→C2→C4 alone reproduces the validated
bubble; C3 first bites at `L≥2` or `d≥2`.

---

## 5. Build vs borrow

- **C0/C1 (Symanzik):** in-house. Our graphs are small and our "masses" are just
  `μ+Dk²` with exponential propagators — the polynomial-from-graph step is short;
  borrow only the *formulas* (Bogner–Weinzierl).
- **C2 (causal time-simplex):** in-house. No external tool models retarded
  `θ`-orderings; this is our physics, and the 1-loop `loop_dyson` time route is
  the template.
- **C3 (sector decomposition):** the algorithm is intricate. Two options:
  **(a)** drive **pySecDec**'s sector-decomposition module on our parametric
  integrand; **(b)** focused in-house implementation tailored to the causal
  heat-kernel structure. **Recommendation:** prototype with (a) to validate the
  math at `d=2` (milestone III.2), then decide whether pySecDec's Euclidean
  assumptions fight the causal structure enough to justify (b). **Decision point
  at III.2.**

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
- **III.2 — turn on `d=2/3`** (the `U^{−d/2}` exponent + `ε`-expansion); first
  test of C3 (sector decomposition). *Oracle:* the known critical-dynamics
  `ε`-expansion of a standard model (Model A/B — Täuber) for the leading poles;
  the static closed forms (`K₀` in `d=2`, Yukawa in `d=3`) for the tree part.
  **Build-vs-borrow decision here.**
- **III.3 — arbitrary `(L,d)`** topologies driven straight from
  `enumerate_unique_diagrams` + `route_momenta`. The general evaluator.

---

## 7. Interfaces / data contracts

Aligns with the §4 integrator interface (the D1/D2 reversal seam):

```
spatial_reduce.reduce(routing: RoutingResult,
                      edge_kinds: dict[edge, {'retarded'|'correlation'}],
                      d: int) -> SymanzikForm
    # SymanzikForm: U(w), F(w,q), prefactor(L,d), edge→duration-parameter semantics
    #   (retarded → fixed Δt; correlation → integrated s ≥ |Δt|)

temporal_integrate.integrate(symanzik: SymanzikForm,
                             orderings,            # Phase-J chambers (causal θ's)
                             ext_times, num_params,
                             backend='C') -> EpsLaurent
    # EpsLaurent: {-p:…, …, 0: finite, …} — poles + renormalized finite part
```

Inputs already exist: `route_momenta` returns `edge_momenta` (linear in
`q_syms`, `loop_syms`) and `n_loops`; `edge_k2()` gives `k_e²`. The
edge-kind tagging comes from the diagram typing (response vs correlation lines).

---

## 8. Risks & open questions (honest)

- **THE open piece:** real-time / causal Symanzik. The Euclidean polynomials are
  textbook; combining them with retarded `θ`-orderings + Keldysh structure is
  *not* a packaged result (architecture §9). C2+C3 are genuine research — budget
  for them being the bulk of the effort.
- **Time-domain vs frequency-domain.** Two ways to set up C2: keep time explicit
  (causal simplex, as above — *recommended*, matches the validated 1-loop), or
  Fourier to `ω` and do `∫dω` with retarded poles by contour. Time-domain avoids
  the contour bookkeeping and inherits the close-pair-free property.
- **Renormalization scheme.** Pin conventions (minimal subtraction; the dynamic
  `Z_φ,Z_D,Z_μ,Z_g,Z_T`) against a reference Model-A/B computation before trusting
  finite parts.
- **Performance.** QMC convergence and sector-count growth at high `L`; mitigated
  by the Lee–Pomeransky single-polynomial form and pySecDec's QMC.
- **Oracles thin out past III.1.** Beyond the simulator-reachable regime,
  cross-checks lean on (i) backend-A agreement, (ii) known `ε`-expansion
  coefficients, (iii) equilibrium Ward identities. Keep at least one live per
  milestone.

---

## 9. References

Same as `docs/spatial_v2_architecture.md` §9: Bogner–Weinzierl
[1002.3458](https://arxiv.org/abs/1002.3458) (Symanzik `U,F`); Weinzierl
[2201.03593](https://arxiv.org/abs/2201.03593) (representations); pySecDec
[2202.13647](https://arxiv.org/abs/2202.13647) (sector decomposition + QMC);
Täuber *Critical Dynamics* + [cond-mat/0511743](https://arxiv.org/abs/cond-mat/0511743)
(MSR-JD dynamic renormalization); Kamenev (Keldysh structure).

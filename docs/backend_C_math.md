# Backend C — mathematical foundation (real-time Symanzik + sector decomposition)

**What this is.** The math that backend C rests on: the parametric (Schwinger)
representation of MSR-JD loop integrals, the Symanzik polynomials in our
heat-kernel time representation, the residual *causal* time-simplex integral, and
its singularity structure. **For finite-scale SPDEs (the core target — §0) a
physical cutoff makes that integral simply finite — no renormalization.** Sector
decomposition + dimensional regularization (turning continuum UV divergences into
a renormalized result) is the *optional* path for critical theories only. The
**momentum reduction is exact at arbitrary loop order `L` and spatial dimension
`d`**; making that practical and renormalized at high `(L,d)` is the job of the
causal time-simplex + sector-decomposition backend — the research component, not
a packaged result.

**Companion docs.** Engineering plan & milestones: `docs/backend_C_design.md`.
High-level decision (backend A now, C later) and reading list:
`docs/spatial_v2_architecture.md` §5 (option C), §9. This note is the detail §9
calls "the research content of the eventual C backend."

**One-line claim.** In the heat-kernel `(k,t)` representation the *edge times are
already Schwinger parameters*, so the loop momentum integral is Gaussian at any L
and any d — it collapses to the Symanzik polynomials `U(w), F(w)` — and the only
genuinely new work is the *causal* time-parameter integral that remains. With a
**physical UV cutoff** (the common case — §0) that integral is plainly finite;
renormalization is needed only in the continuum-critical limit.

---

## 0. Regimes — when renormalization is needed (and when it is not)

**Renormalization is NOT a prerequisite for backend C.** Which machinery you need
depends on the regime, and the regime most relevant here (finite-scale SPDEs:
neural fields, lattice/RDME simulators, finite synaptic/axonal ranges, colored
noise) needs *none* of it.

- **Regime 1 — effective finite-scale SPDE (the core target).** The system has a
  *real* UV cutoff: a spatial mesh `a`, a finite synaptic/axonal range, a finite
  noise-correlation length, a smooth connectivity kernel `Ĵ(k)`. Loop integrals
  are cut off at `|k|≲1/a` (or smoothly by `Ĵ`) and are simply **finite** — you
  compute the finite correction, no infinities, no renormalization. This is the
  honest description of cortex / of any grid simulation (neither is continuum QFT
  to arbitrarily small scales) and is standard in statistical mechanics.
- **Regime 2 — finite observable from a formally-divergent loop.** Even in the
  continuum the quantity of interest is often finite by a single subtraction:
  `⟨φ(x)φ(y)⟩` at `|x−y|>0` is finite though `⟨φ(x)²⟩` is not; `Σ(q)−Σ(0)` is
  finite though `Σ(0)` is not. If the science is "how do correlations shift?"
  (not "what are the universal exponents?"), one subtraction suffices — **no
  renormalization machinery.** *Our session's conserved `∇²(φ²)` bubble is already
  a Regime-2 result: the `q²` form factor IS `Σ(q)−Σ(0)` with `Σ(0)=0`, finite as
  it stands.*
- **Regime 3 — true critical continuum theory.** Renormalization is unavoidable
  only for `Λ→∞`, near criticality (`μ→0`, diverging correlation length), with
  universal scaling exponents — KPZ, Model A/B criticality, Wilson–Fisher. Here
  the `ε`-poles and RG *are* the physics.

**Scope decision (the strategic simplification).** The core backend-C engine
targets **Regimes 1 & 2: automated perturbative MSR-JD for finite-scale SPDEs
with physically meaningful cutoffs.** Renormalization (sector decomposition +
dim-reg + RG, §5) is an **optional module for Regime 3, off the critical path.**
This removes the hardest research piece from the core, keeps every core result
directly comparable to a simulation at the *same* cutoff, and is the more relevant
target for neuroscience/SPDE applications.

---

## 1. The objects we integrate

A typed MSR-JD diagram has internal vertices at times `t₁…t_V`, joined by two
edge types (with edge momentum `k_e` fixed by `route_momenta`, a linear form in
the external `q_j` and loop `ℓ_i`):

- **retarded** `G_R(k_e, Δt) = θ(Δt) · e^{−(μ+D k_e²)Δt}`  — causal, carries an ordering `Δt>0`;
- **correlation** `C(k_e, Δt) = (T/m_{k_e}) · e^{−m_{k_e}|Δt|}`, `m_k = μ+Dk²`.

The single structural fact that makes C work: **both edge types are
`e^{−(μ+Dk_e²)·w_e}` for a non-negative "duration" `w_e`.** For a retarded edge
`w_e = Δt_e` is fixed by the vertex times. For a correlation edge, use
`1/m_k = ∫₀^∞ dσ e^{−m_k σ}` to write

```
C(k,Δt) = (T/m_k) e^{−m_k|Δt|} = T ∫_{|Δt|}^∞ ds  e^{−m_k s},
```

so a correlation edge carries a Schwinger parameter `w_e = s_e ≥ |Δt_e|` that is
*integrated*. (This is exactly what `loop_parametric.sigma_R_kernel` already does:
the correlation edge weight runs `s ∈ [t, ∞)`.) In every case the
`k`-dependence is `e^{−D k_e² w_e}` and the mass part factors out as
`e^{−μ Σ_e w_e}`.

**Scope of the base case.** The OU/equilibrium form `C=(T/m_k)e^{−m_k|Δt|}` above
is the *single-pole* base case. Multi-pole or Markovian-embedded **colored**
noise, and **multi-field** systems, give a correlation edge that is a **finite
sum over modes** — each term with its own Schwinger parameter `w_e`, residue, and
mass `m_{a,k}` (matrix-valued across fields). Backend C must treat a correlation
edge as such a sum, not hard-code the single-pole form; the Gaussian momentum
reduction below applies term-by-term, so this is a bookkeeping generalization,
not a new integral.

---

## 2. The key identity — times are Schwinger parameters

Write each edge momentum from `route_momenta` as
`k_e = Σ_i a_{ei} ℓ_i + Σ_j b_{ej} q_j` (loop coefficients `a_{ei}`, external
coefficients `b_{ej}`). The loop-momentum integral over the diagram is then a
pure Gaussian in the `L` loop momenta:

```
I_mom(w,q) = ∫ ∏_{i=1}^{L} d^dℓ_i/(2π)^d  exp[ −D Σ_e w_e k_e² ]
           = ∫ ∏ d^dℓ/(2π)^d  exp[ −D ( ℓᵀ M(w) ℓ + 2 ℓᵀ N(w) q + qᵀ Q(w) q ) ]
```

with the `L×L`, `L×(k−1)`, `(k−1)×(k−1)` matrices

```
M_{ii'} = Σ_e w_e a_{ei} a_{ei'},   N_{ij} = Σ_e w_e a_{ei} b_{ej},   Q_{jj'} = Σ_e w_e b_{ej} b_{ej'}.
```

Doing the Gaussian (each of the `d` spatial components contributes one factor):

```
I_mom(w,q) = (4πD)^{−Ld/2} · U(w)^{−d/2} · exp[ −D · F(w,q)/U(w) ],
   U(w) = det M(w)                    (first Symanzik polynomial),
   F(w,q)/U(w) = qᵀ ( Q − Nᵀ M⁻¹ N ) q   (second Symanzik form).
```

`U` is the sum over spanning trees `Σ_{T} ∏_{e∉T} w_e`; `F` is the sum over
spanning 2-forests weighted by the squared momentum crossing the cut
(Bogner–Weinzierl, arXiv:1002.3458). **`d` enters only as the exponent `−d/2` on
`U`** — which is precisely why "any d" is a parameter flip, not a re-derivation.

**This generalizes code we already have and validated.** `loop_parametric.
gaussian_momentum_integral(a,b,w,q,D,spatial_dim)` is the `L=1` case verbatim:
`U=Σ a_e² w_e`, `V=Σ a_e b_e w_e`, `W=Σ b_e² w_e`, `F=W−V²/U`, prefactor
`(4πDU)^{−d/2}`, result `pref·e^{−Dq²F}`. Backend C's C0/C1 are the promotion of
the scalars `(U,V,W)` to the matrices `(M,N,Q)` and `det/inverse` — nothing
conceptually new in the momentum step.

---

## 3. What remains — the causal time-simplex (the MSR-JD-specific part)

After `I_mom`, the diagram value is a finite-dimensional integral over the edge
durations / vertex times:

```
Γ(q, τ_ext) = M(Γ)·(couplings, T's) · ∫_{Δ} ∏ dw  e^{−μ Σ_e w_e} · U(w)^{−d/2} · e^{−D F(w,q)/U(w)}
```

over a domain `Δ` set by the **causal structure**:

- each **retarded** edge imposes an ordering `Δt_e > 0` on its endpoint vertex
  times — together these carve the vertex-time space into **ordering chambers**
  (the Phase-J polytope; backend A already enumerates these for any topology);
- each **correlation** edge contributes a Schwinger integral `s_e ∈ [|Δt_e|, ∞)`;
- the **external lag(s) `τ_ext`** enter as fixed boundary data (the times of the
  external legs).

This causal, chambered domain is what distinguishes our integrals from textbook
*Euclidean* Feynman integrals (which are fully symmetric, no `θ`'s). It is the
same object backend A integrates per chamber by quadrature and backend B
(`loop_dyson`) does by explicit convolution for the bubble — **C's contribution
is to do it after the Symanzik reduction (smooth in momentum), handle the genuine
UV singularities systematically, and avoid the close-pair pathology by
construction (§4–5).**

---

## 4. Singularity structure — one genuine divergence, one representation hazard

These are different *in kind* and must not be conflated. (a) is a *continuum*
divergence that a **physical cutoff removes** (Regimes 1–2, §0) — only the
continuum-critical limit (Regime 3) needs sector decomposition; (b) is a
numerical cancellation introduced by a *choice of representation*, avoided by not
making that choice.

**(a) UV — a continuum-limit divergence the cutoff removes.** Only in the
*continuum* limit `Λ→∞`: `U(w)` is homogeneous of degree `L`, so as all `w_e→0`
together `U→0` and `U^{−d/2}→∞`. Under uniform scaling the local integral behaves
like `∫ w^{(#edges)−1−Ld/2} dw` — the **superficial** degree of divergence
(worsening with `d, L`); the full structure also has **subdivergences** (only a
*subgraph*'s `w_e→0`), which in the continuum need **forest/sector decomposition**
(§5, the optional Regime-3 module). **With a physical cutoff this whole region is
truncated and the integral is finite — you just compute it.** How the cutoff
enters the Symanzik form:

- **smooth Gaussian** (a connectivity kernel `Ĵ(k)=e^{−σ²k²}` or a regulator
  `e^{−σ²k_e²}` per edge) — *the friendliest case.* It adds `σ²/D` to each edge
  weight, `w_e → w_e + σ²/D`, so the momentum integral **stays closed-form
  Gaussian** AND the weights never reach 0: `U(w)` is bounded below, so
  `U^{−d/2}` is finite and the singularity **never arises**. (The `σ_R(a)~a^{−1/2}`
  sliver fixed this session is needed *only* in the strict `σ→0` continuum limit.)
- **hard cutoff** `|k|<Λ` — the momentum Gaussian becomes an incomplete Gaussian
  (`erf` / incomplete-Γ) or a numerical radial integral. Finite, slightly less
  clean than the smooth case.
- **lattice / mesh** (finite `a`) — the momentum integral runs over the Brillouin
  zone `[−π/a, π/a]^d` with the lattice dispersion (`m_k = μ + (2D/a²)Σ_i(1−cos
  k_i a)`). Finite, and **most faithful to a grid simulator** (the simulator *is*
  this lattice theory).

**Match the cutoff to the simulator.** A grid simulator with `N` points over
length `L` has `k_max = πN/L`; using that *same* cutoff in the loop integral is
what makes theory-vs-sim agreement a genuine test, not a fudge. (The bubble code's
`q_cut` is exactly this — the spatial engine already runs in Regime 1.)

**(b) Close-pair — NOT a divergence of this integral; a representation artifact.**
The close-pair pathology `(e^{−mt}−e^{−m't})/(m−m')` is a *numerical cancellation*
introduced by **partial-fractioning** a time integral into pole-difference
denominators — it is not an endpoint/boundary singularity of the original
parametric integral. The parametric representation **avoids it structurally: the
loop integration never forms `1/(λᵢ−λⱼ)` denominators in the first place.**
Sector decomposition is *not* the cure here — it addresses endpoint singularities,
and helps with a near-degeneracy only if that degeneracy happens to map to a
sector boundary. If a *later* analytic reduction reintroduces ratios such as
`(e^{−mt}−e^{−m't})/(m−m')`, those must be evaluated with stable **divided-
difference / repeated-pole (confluent)** routines — never by computing `m−m'` and
dividing.

---

## 5. Sector decomposition + dimensional regularization — OPTIONAL (Regime 3 only)

**This section is the optional Regime-3 module, NOT part of the core engine.** It
is needed only for the continuum-critical limit (`Λ→∞`, `μ→0`, universal
exponents). For finite-scale SPDEs (Regimes 1–2) the cutoff in §4(a) already makes
the `w`-integral finite, and the core integrator (C3-lite) is just robust
adaptive quadrature on that finite integral — no `ε`-poles, no subtraction. Read
on only if you want critical exponents / continuum universality.

Sector decomposition (Hepp; Binoth–Heinrich; implemented in pySecDec,
arXiv:2202.13647) makes the *continuum* `w`-integral finite and numerically
tractable:

1. **Split** the domain into sectors that disentangle overlapping singularities;
   remap each sector to the unit hypercube `[0,1]^n`.
2. **Factorize** the singular behaviour in each sector as a monomial
   `∏_i x_i^{−1+ε aᵢ}` by setting `d = d_c − 2ε` around a critical dimension
   `d_c`. The divergence becomes an explicit pole in `ε`.
3. **Extract** the `1/ε^k` poles (subtraction), leaving finite integrands.
4. **Evaluate** the finite unit-cube integrals numerically (quasi-Monte-Carlo).

The output is a Laurent series in `ε`:
`Γ = c_{−p}/ε^p + … + c_{−1}/ε + c_0 + O(ε)`.

**Renormalization.** The poles `c_{−k}` are the UV divergences; they are absorbed
into renormalizations of the bare parameters `(μ, D, couplings, T)` — i.e. the
dynamic renormalization constants `Z_φ, Z_D, Z_μ, Z_g, Z_T` (minimal
subtraction). For dynamical field theories these `Z`'s *are* the physics (the RG
flow, dynamic exponent `z`, critical dimensions — Täuber, *Critical Dynamics*).
The finite part `c_0` is the renormalized correlator. Equilibrium models
(detailed balance) give Ward-like relations among the `Z`'s that serve as
**internal consistency checks**.

---

## 6. Worked anchors (the validation oracles)

**1-loop bubble (reaction-diffusion `φ̃φ²`)** — `L=1`, two edges
`G_R(ℓ), C(q−ℓ)` ⇒ `a=(1,−1)`, `b=(0,1)`:
`U = w₁+w₂`, `F = q² w₁w₂/(w₁+w₂)`,
`I_mom = (4πD(w₁+w₂))^{−d/2} e^{−Dq² w₁w₂/(w₁+w₂)}`.
The remaining `∫dw` over the two durations reproduces `Σ_R(q,t)` — *bit-for-bit
what `gaussian_momentum_integral` + the `loop_parametric` integrand already
compute*. ⇒ milestone **III.0** uses this as a known-good oracle (backend B and
the simulator).

**2-loop sunset (`φ̃φ²`)** — `L=2`, three correlation edges
`C(ℓ₁)C(ℓ₂)C(q−ℓ₁−ℓ₂)`:
`U = w₁w₂ + w₂w₃ + w₃w₁` (the `2×2` `det M`),
`F = q² w₁w₂w₃ / U`,
`I_mom = (4πD)^{−d} U^{−d/2} e^{−Dq²F/U}`.
The remaining `∫dw₁dw₂dw₃` over the (causally ordered) durations is the first
genuinely new evaluation ⇒ milestone **III.1**, validated against a brute-force
`∫dℓ₁dℓ₂` and the simulator.

---

## 7. What the math does and does not cover

**Covers:** the loop evaluation. The **momentum reduction (Symanzik) is exact at
arbitrary `L`, `d`** and mature; with a physical cutoff (Regimes 1–2) the residual
parametric integral is finite, so the **core** engine is just the causal
time-simplex (§3) + robust adaptive quadrature, plus **structural avoidance** of
the close-pair pathology (no pole-difference denominators — §4b). The genuinely
new work in the core is the *causal* time-simplex integral (§3) adapted to
retarded/Keldysh propagators — engineering, not renormalization. Sector
decomposition + dim-reg (§5) is the **optional Regime-3 module** for the
continuum-critical limit; that real-time-Symanzik adaptation is the only
research-grade piece, and it is off the core critical path.

**Does not cover** (handled elsewhere): the `k>2` external multi-momentum →
multi-position output Fourier transform (a different integral — the *external*
transform, not the loop); the non-Gaussian-noise *vertex content* (authoring +
enumeration; already generic for local cumulants). These compose with C but are
separate workstreams.

---

## References

- C. Bogner & S. Weinzierl, *Feynman graph polynomials*, IJMPA 25 (2010) 2585,
  [arXiv:1002.3458](https://arxiv.org/abs/1002.3458) — the canonical `U`, `F`
  Symanzik polynomials read off a graph.
- S. Weinzierl, *Feynman Integrals* (Springer, 2022),
  [arXiv:2201.03593](https://arxiv.org/abs/2201.03593) — Feynman/Schwinger/
  Lee–Pomeransky representations; the Lee–Pomeransky single-polynomial form is
  cleanest for numerics.
- S. Borowka et al., *pySecDec*,
  [arXiv:2202.13647](https://arxiv.org/abs/2202.13647),
  [arXiv:2311.00492](https://arxiv.org/abs/2311.00492) — sector decomposition +
  QMC; the practical engine.
- U. C. Täuber, *Critical Dynamics* (Cambridge, 2014) and
  [arXiv:cond-mat/0511743](https://arxiv.org/abs/cond-mat/0511743) — the MSR-JD
  response functional and how *dynamic* loop integrals / renormalization are
  organized (closest to our setting).
- A. Kamenev, *Field Theory of Non-Equilibrium Systems* (Cambridge) — the
  Keldysh/closed-time-path structure the parametric rep must be adapted to.

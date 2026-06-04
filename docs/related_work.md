# Related work & novelty map

*Branch `spatial-extension`, June 2026.* Where Daedalus sits relative to the literature,
from two adversarially-verified deep-research passes (one on the end-to-end pipeline, one on
the Appendix-A enumeration bounds). Verdicts split into **standard → must cite** vs
**appears novel**. Novelty findings rest on absence-of-evidence within the surveyed corpus
(medium confidence); the residual gaps to close before publication are listed at the end.

## Model class (how to name it)
The framework targets **semilinear Langevin SPDEs** — MSR–JD response-field theories with a
constant-coefficient diffusive linear part (`∂_t + μ − D∇²` → heat-kernel propagator),
local polynomial field/derivative nonlinearities, and Gaussian (white or `∇²`-conserved)
noise. In physics terms: the class spanning the **Hohenberg–Halperin Models A/B/…** and the
**KPZ–Burgers** universality classes. (Avoid "nonlinear diffusion" — that means state-
dependent `D(φ)`, which the constant-`D` propagator excludes.)

## A. The end-to-end pipeline (action → diagrams → Symanzik/heat-kernel/Bessel/MC)

**Closest prior art.**
- **Pereira 2022**, *Generating and evaluating 1PI Feynman diagrams for an MSR field theory*
  (arXiv:2208.09040). Retrofits **FeynArts** per model (hand-written model files; the
  physical/response split faked via a repurposed `Mass=1/0` attribute; mixing fields hand-
  defined; internal `Analytic.m` patched), evaluates via **SecDec** dim-reg + ε. *One model
  (NSAPS); not a generic-action tool.* Same two-stage `CreateTopologies → InsertFields`
  enumeration split as our prediagram → typed-diagram, but per-model and momentum-space.
- **Adzhemyan, Davletbaeva, Evdokimov, Kompaniets 2025**, *Multiloop calculations with
  parametric integration in critical dynamics: four-loop Model A* (arXiv:2512.10591); + the
  group's Model-A 4–5-loop (arXiv:1712.05917; Phys.Lett.A 425 (2022) 127870). Computes the
  **universal exponent z** (ε-series) by **"time versions" (= our causal chambers) →
  integrate times analytically into energy denominators → momentum-space parametric
  integration with hyperlogarithms (HyperInt)**. Massless, renormalized, analytic.
- **Hnatič et al. 2026**, directed percolation 3-loop via **sector decomposition + Vegas MC**
  (arXiv:2602.11369). MC/SD for MSR-JD loops — momentum space.

**The "dual strategy" framing (key for positioning vs Adzhemyan).** After the shared
time-ordered decomposition, the two approaches transpose: *they* integrate **times**
analytically and do the **momentum** integral parametrically (→ universal exponents,
momentum space, massless, dim-reg, fully analytic); *we* integrate the **momentum**
analytically (Symanzik → heat-kernel real-space IFT) and do the **time/σ** integral
numerically (→ finite off-critical correlators `C(x,τ)` at physical `d`, model-independent,
validated vs simulation). Different observable, regime, and method — complementary, not
competing.

**Standard → must cite.** Automated MSR diagram generation exists (Pereira); the parametric
/ Symanzik representation in dynamical FT exists (Adzhemyan/Kompaniets); SD + MC for
dynamical loops exists (Hnatič, Adzhemyan); the cosmological-LSS↔QFT-integral correspondence
(Simonović–Baldauf, arXiv:1708.08130) is a cognate parametric loop program. The canonical
MSR-JD evaluation (Täuber, *Critical Dynamics*, CUP 2014; Lect.Notes Phys. 716) does ω-
residue + dim-reg Γ-masters — **no** Schwinger/Symanzik/heat-kernel/Bessel/SD/MC.

**Appears novel (the combination + two links).** The **real-space heat-kernel inverse-FT +
Bessel-K radial reduction** of dynamical loop integrals (found in none of the surveyed
literature); the **model-independent generic-action** engine with native MSR response/causal
structure; and the **causal-chamber → Symanzik (momentum-analytic-first)** route. The
"causal-chamber/time-version" decomposition itself is **not** novel — it is the
Adzhemyan-school "time versions" technique (and appears in the Ocker–Buice lineage); cite it.

## B. The Appendix-A pre-diagram enumeration bounds

**Standard → must cite** (the bed the proofs rest on):
- diagram **counting** by loop order (zero-dim field theory / generating functions): total/
  asymptotic counts, never structural per-vertex bounds — Cvitanović–Lautrup–Pearson,
  Phys. Rev. D 18 (1978) 1939; Borinsky, Ann. Phys. 385 (2017) 95 + thesis arXiv:1807.02046.
- **spanning-tree + loop-edges (cycle-space/matroid)** decomposition, **Symanzik U/F**, and
  `L = I − V + 1 = b₁` — Bogner–Weinzierl, IJMPA 25 (2010) 2585; Weinzierl, arXiv:1301.6918.
- the **"genuine vertex has valency ≥ 3 / no degree-2 chains"** 1PI/skeleton convention —
  Weinzierl 2013; Cvitanović et al. 1978 (skeleton counting).
- algorithmic generators' **operational** completeness (no a-priori bound theorems; typing/
  direction post-hoc) — FeynArts (Hahn, CPC 140 (2001) 418; Küblbeck–Böhm–Denner 1990);
  GraphState/Graphine Nickel-index, arXiv:1409.8227 (the closest directed+typed
  representation — but colour-typing ⟂ degree, and direction admits cycles).
- the **directed-because-causal** observation for stochastic diagrammatics — **Ocker, Josić,
  Shea-Brown & Buice, PLoS Comput. Biol. 13 (2017) e1005583** ("the graphs are directed,
  since we only consider causal systems where measured cumulants are only influenced by past
  events") + loop power-counting `m = n + l − 1`; Helias–Dahmen, arXiv:1901.10416 (derives
  `L = I − V + 1` via the tree+extra-edges construction). **This is the direct lineage —
  cite heavily.**

**Appears genuinely new.** No located source states either: (i) the **axiomatization** of
pre-diagrams as *acyclic directed typed multigraphs with degree-determined vertex types*
(ext = in-deg 1/out-deg 0; src = in-deg 0; int = else), or (ii) the **explicit `(ℓ,k)`
completeness bounds** — `j ≤ ℓ + ⌊ℓ/2⌋` (leaves), `≤ k+j−2` (branching), the degree-2 bound
— giving a provably finite, complete, a-priori-pruned enumeration. The Ocker–Buice lineage
has the causal-directed *idea* but never formalizes the acyclic directed typed multigraph,
never types by in/out-degree, and proves no enumeration/finiteness bound. The delta over the
authors' own program is the **formalization + the bounds**.

**Suggested paper sentence.** *"Pre-diagrams are formalized as acyclic directed typed
multigraphs; building on the standard spanning-tree/cycle-space decomposition
[Bogner–Weinzierl; Weinzierl], the `L=I−V+1` loop relation, the 1PI-skeleton convention, and
the directed-because-causal diagrammatics of spiking-network field theory [Ocker et al. 2017;
Helias–Dahmen], we prove explicit `(ℓ,k)` bounds (Thm A.28) yielding a provably complete,
finite enumeration — to our knowledge the first such completeness/finiteness result for
causal stochastic field-theory diagrams."*

## Residual gaps to close before publication
1. **qgraf** (Nogueira, CPC 105 (1997) 279) — named but not directly read; it handles
   directed propagators natively, so check its completeness statement / directedness.
2. **Pure graph-combinatorics**: is there an existing inequality bounding leaves/branching
   vertices by the first Betti (cyclomatic) number — the most likely home of an analog to
   `j ≤ ℓ + ⌊ℓ/2⌋` (the "graphs with given cyclomatic number" / rooted-tree+excess-edge
   literature)?
3. **Doi–Peliti reaction-diffusion / chemical-master-equation** enumeration — not directly
   surveyed; a directed-structured enumeration there is a residual unknown.
4. **Later Ocker/Buice/Doiron (2018–2026)** — self-check whether any already states the
   vertex-type-by-degree definitions or an `(ℓ,k)` bound (would move part to "incremental
   within our own program").

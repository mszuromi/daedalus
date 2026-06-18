# Holistic readability brief — novice-reader lens

**Reviewer persona:** a physicist fluent in MSR-JD field theory, SDEs, Feynman
diagrams, and the loop expansion, but with **zero** prior exposure to SageMath,
nauty, numba, SymPy, or the project's own conventions. The manual's own preface
(`00-how-to-read.tex`) makes two explicit promises I hold it to:

1. external tooling is explained "from the ground up" / "when in doubt we
   over-explain";
2. the **first** appearance of a technical term is bolded (`\term{}`) and
   defined nearby, and also lives in the glossary (App. A).

Chapters read end-to-end: **01-overview, 02-quickstart, 09-enumeration,
12-phasej-temporal, 14-spatial-core.**

Chapter ordering (for judging forward refs): Part I = ch1, ch2; Part II = ch3–5;
Part III = ch6 fieldtheory, ch7 propagator, ch8 mean-field, **ch9 enumeration**,
ch10 type-assignment, ch11 caching; Part IV = **ch12 phasej**, ch13 grouped;
Part V = **ch14 spatial-core**, ch15 heatkernel, ch16 coupled; App. A glossary,
B external-tools, C file-index, D known-issues.

The four failure modes I hunted for: (a) jargon used before defined; (b) a
forward-reference to something not yet explained; (c) a derivation step a
newcomer needs but that is skipped; (d) a code listing shown without telling the
reader what to look at.

Overall the chapters are unusually good for a newcomer — nauty, SymPy, numba,
einsum, dataclass, fork-vs-spawn are all introduced from scratch exactly as
promised. The findings below are the specific spots where the promise slips.

---

## CRITICAL — none

No spot is so broken it blocks comprehension outright. The items below are all
real but recoverable with a clause or a pointer.

---

## MAJOR

### M1. Ch12 §sec:modesum — the contour-closing derivation skips the one step a newcomer needs (jargon/derivation gap)
`12-phasej-temporal.tex` line ~143: the inverse Fourier transform is "computed
by closing the ω contour and summing residues. The causality filter ... every
pole p_α ... lies in the upper half plane." For a reader who knows MSR-JD but is
rusty on contour integration, the **load-bearing** missing step is *why the
semicircular arc contributes nothing and which half-plane you must close in*.
The sign of `t` in `e^{i p t}` dictates closing up (t>0) vs down (t<0), and that
choice is *exactly* what produces the Θ(Δt) retardation the whole chapter rests
on. Right now the Heaviside appears one line later by assertion ("multiplied by a
Heaviside that kills the growing t<0 half") with no link to the contour choice.
Add one sentence: "for Δt>0 we close in the upper half-plane, where the arc
decays (Jordan's lemma); for Δt<0 the upper contour encloses no poles, giving 0
— which is the Heaviside." This converts an asserted result into a derived one
and is the single highest-value fix in the temporal chapter.

### M2. Ch12 §sec:polytope — "free external times t_free" is used ~70 lines before "free" is explained (jargon before defined)
`12-phasej-temporal.tex` line 323 (eq. `dt-affine`) introduces "the free external
times t_free" as if the reader already knows which external times are "free." The
concept that one external leaf is **pinned to t=0** (origin pinning) and the
*rest* are therefore "free" is only explained in §sec:wick "Origin pinning" at
lines ~390–399, well after. A newcomer hits `t_free` cold. Fix: either move a
one-line forward-pointer at line 323 ("free = the external times other than the
pinned origin leaf; see §origin-pinning") or, better, introduce origin-pinning
before the affine-Δt subsection so the term is defined on first use. The same
unexplained "free" recurs in the `EdgeModeSum` field comment
(`dt_ext_pairs ... free external times`, line 202) and the re-pinning listing
(line 458 "subtract t_origin from the free legs").

### M3. Ch14 worked-bubble example invokes the heat-kernel formula it has explicitly deferred (forward reference to undefined object)
`14-spatial-core.tex` §"A worked shape: the one-loop bubble" (lines 711–722) ends
the chapter's capstone concrete example with: "the heat kernel
`(4πB)^{-1/2} e^{-x²/4B}` is summed over the grid into δC(x) (the analytic IFT of
Chapter 15)." But the chapter has **not** derived this real-space heat-kernel
form, nor defined `B = D·Q_eff` as a *width* — `B` first appears literally one
line above (718) with no statement that it plays the role of a diffusion
width/variance, and the IFT that turns the q-Gaussian into `e^{-x²/4B}` is the
subject of the *next* chapter. So the reader is asked to accept the chapter's
climactic formula on faith with a pointer forward. This is defensible as a
teaser, but as written it reads as a gap. Minimum fix: add a half-sentence naming
B ("B = D·Q_eff plays the role of the diffusion width") and flag explicitly "we
quote the real-space kernel here; it is derived in Ch15," so the forward
dependence is honest rather than implicit.

### M4. Ch14 — "self-energy σ(q,u)" and the Σ_R+Σ_A / Keldysh machinery appear undefined and with clashing notation (jargon before defined)
`14-spatial-core.tex`: line 847 writes the retarded+advanced completion as
"Σ_R(τ)+Σ_A(τ) = Γ(τ)+Γ(-τ)" and tags `{R,R}` as "(Keldysh)" with no definition;
line 891 (oracle section) refers to "self-energy σ(q,u)." Three distinct problems
for a newcomer: (i) **Keldysh** is never defined in this chapter — it is in the
glossary, but there is no pointer, violating the first-use-defined promise;
(ii) the symbol **Σ** (capital, for the retarded/advanced insertion) and **σ**
(lower-case, for the oracle's self-energy) are introduced in the same chapter for
related-but-different objects with no disambiguation; (iii) "self-energy" itself
is standard physics (OK for the audience) but the chapter never says the
insertion Γ being completed *is* a self-energy-type object, so the link between
§"Assembling a correlator" and the oracle's σ(q,u) is invisible. Fix: one
sentence at line 847 defining the R/A (retarded/advanced) and Keldysh {R,R}
labels with a glossary pointer, and a note that capital Σ here is the diagram's
insertion, distinct from the oracle's self-energy σ.

---

## MINOR

### m1. Ch2 §"seven-phase trace" — "bigrade" used 4 chapters before its definition (jargon before defined)
`02-quickstart.tex` line 430: phase [1/7] "classify every term by its 'bigrade'
(how many response and physical field factors it carries)." This is the *only*
occurrence of bigrade in ch2, and the parenthetical gloss is actually decent —
but the formal `\term{bigrade}` definition is in `06-fieldtheory-core.tex`
§sec:bigrade (line ~608). A linear reader meets a scare-quoted term 4 chapters
early. The gloss mostly saves it; the clean fix is a parenthetical pointer
"(defined in the field-theory-core chapter)" matching how the same sentence
already points other phases to their chapters.

### m2. Ch9 §"three-stage refinement" forward-references "MSR-JD structural constraints" before the section that lists them (forward reference, in-chapter)
`09-enumeration.tex` line 136: topologies keep "only those that satisfy MSR-JD's
structural constraints" — but the actual constraint list
(`check_topology_constraints`, the no-adjacent-deg-2 / isolated-hub / leaf rules)
is §"The structural constraints on a topology" at line 477, 340 lines later. This
is a same-chapter forward ref so it is mild, but the phrase is doing real work in
the overview and a reader can't tell whether "structural constraints" means
something they should already know. A `(\S\ref{...})` pointer fixes it.

### m3. Ch9 §sec:iso mis-cited as the home of the *causality* constraints (forward/cross reference points to the wrong place)
`09-enumeration.tex`: the worked walk-through (line 1126) says "Orienting those 9
topologies under the causality constraints of §sec:iso" — but §sec:iso is "Graph
isomorphism and nauty," which is about *dedup*, not orientation. The causality
(orientation) constraints are in §"The MSR-JD causality constraints"
(`check_orientation_constraints`, line ~621). A newcomer who follows the
cross-ref lands on the wrong section. Fix the `\ref` target to the
orientation-constraints subsection.

### m4. Ch12 §sec:modesum — the [p,r] transpose is asserted as a fact to "respect" with the explanation outsourced (derivation gap + forward ref)
`12-phasej-temporal.tex` lines 161–170: "There is a transpose to watch here: the
kernel matrix K ... is laid out [resp, phys], so to read off a retarded entry you
index G[phys, resp]. The propagator chapter covers this; Phase J just respects
it." For a reader who jumped to Part IV (which the preface explicitly blesses),
"the propagator chapter covers this" is a bare back-reference to ch7 with no
inline reason. Index ordering bugs are exactly where a newcomer extending the
code will trip. A one-line "because G = K^{-1} and inversion swaps the row/column
roles" would make the transpose self-contained.

### m5. Ch14 §sec:descriptor — `fpairs`/"Dyson 3c" in the CEdge listing shown with no in-chapter referent (code listing without saying what to look at)
`14-spatial-core.tex` lines 532–542: the `CEdge` dataclass listing carries
`fpairs: tuple = ()  # coupled-field propagator matrix indices (Dyson 3c)`. The
walk-through of `diagram_to_cstack` that follows never mentions `fpairs`, and
"Dyson 3c" is an internal phase label defined only in ch16 (coupled multi-field).
A reader studying the canonical edge record sees a field with a cryptic tag and
no thread to pull. Either drop `fpairs` from the listing shown here (it belongs
to ch16) or add "—ignore until the coupled-field chapter (Ch16); for single-field
diagrams it is always empty."

### m6. Ch14 §"three backends" — "form-factor vertices" treated as known before the heat-kernel chapter defines them (jargon before defined)
`14-spatial-core.tex` lines 814–822 (the `'mc'` gotcha): "biased for derivative
(form-factor) vertices: the detΛ→0 singularity gives the Monte-Carlo estimator
infinite variance." "Form factor" is parenthetically equated to "derivative" but
never *defined* in this chapter — its actual treatment (∂_x → i k, the polynomial
form factor read off the vertex) is Ch15. The reader is told a backend is biased
for an object-class they haven't met. A one-clause gloss ("a vertex carrying
spatial derivatives, e.g. KPZ's (∇h)²; see Ch15") would close it.

### m7. Ch9 §sec:bounds — the completeness-bound proof is summarized as three ingredients then sent to "the paper appendix" (derivation gap)
`09-enumeration.tex` lines 304–319: the three ingredients of the
k+3ℓ−j−1 bound (degree-partition identity; orientability constraint
|V₂^G|≤k+ℓ−1; incidence bound θ≤2ℓ−j) are *stated* but two of the three
(orientability ceiling, the θ incidence count) are asserted with no sketch of
*why*, and the combination step "Combining the three yields (eq v2max)" is a
black box — the actual algebra combining them is not shown, only the separate
substitution check for `v2_max` vs the bound. For a reader who wants to trust
"provably complete," this is the one place the manual says "trust the paper." It
may be an acceptable scope cut, but it is the largest single derivation skip in
ch9 and worth a one-line "(full proof: paper App. X; the combination is a linear
elimination of θ and |V₂^G|)" so the reader knows what's being deferred and that
it *is* elementary algebra, not a deep lemma.

### m8. Ch14 §"Where this sits" data-flow box uses `2^{-n_C}`, `S(Γ)`, `n_C` before they're defined (jargon before defined, in-chapter)
`14-spatial-core.tex` line 113 (the ASCII flow box): "sum over diagrams,
× 2^{-n_C} . S(Gamma) . prefactor." At this point `n_C` (number of C edges),
`S(Γ)` (symmetry factor), and the `2^{-n_C}` convention are all undefined — `n_C`
and `2^{-n_C}` are explained only in §sec:norm (lines 451–467), and S(Γ) is
assumed from upstream chapters. The box is an at-a-glance map so some forward use
is inevitable, but `2^{-n_C}` in particular is a non-obvious normalization that a
newcomer cannot parse here. A footnote on the box ("2^{-n_C} and S(Γ) are defined
in §Normalization below") would suffice.

---

## NITS

- **n1. Ch1 line ~96**: `compute_cumulants` is described as living in
  `pipeline/compute.py` and wrapped by `dd.run`; the seven-phase list is then in
  ch2. Fine, but ch1 never says the verb the user actually types is `dd.run` (it
  says compute_cumulants is "wrapped for users by dd.run") — a first-time reader
  may not realize ch2's `dd.run` *is* this. Trivial.
- **n2. Ch9 §sec:itertools / §sec:parallel etc.**: the per-tool subsections are
  excellent, but `concurrent.futures` is introduced (line ~888) *after* the code
  that uses `ThreadPoolExecutor` (Stage-2 listings) already appeared — the
  reader meets the thread pool in `process_tree_parallel` discussion before the
  primer. Mild ordering wrinkle; the gotcha box at line 903 mostly compensates.
- **n3. Ch12 line ~108**: "the notebook-facing callables total_C and
  total_C_batch" are named as outputs before §sec:parallel (line 847) explains
  them. A reader meets the names twice; harmless.
- **n4. Ch14 line 718**: "P ≈ 22²·24²" uses the literal default node counts
  (n_t=22, n_s=24) from the `diagram_kinematic` signature without restating them
  at the example, so the reader must remember them from line 612. One-clause
  reminder would help.
- **n5. Ch14 §sec:norm** writes the noise vertex as both "−T φ̃²" and
  "2T φ̃ φ̃ in the symmetric normalization" (line 159) and later "the 2T
  noise-vertex convention" (line 454). The factor-of-2 bookkeeping is correct but
  the reader sees three spellings of the same vertex; a single canonical
  statement up front would reduce friction.

---

## What works well (so the rewrite doesn't regress it)

- nauty introduced from first principles incl. the "No AUTomorphisms, Yes?"
  etymology and the canonical-form mechanism (ch9 §sec:iso) — exemplary.
- SymPy, numba (+`@njit`, `cache=True`), einsum, `@dataclass`/`frozen=True`,
  fork-vs-spawn, Sutherland–Hodgman, Schur complement, Schwinger parameter,
  linear extension/poset — every one is defined on first use with a `\defn` or
  inline gloss. This is the standard the four MAJOR findings fall short of.
- The ch2 worked example's "four verbs" spine (load_theory → Config → run →
  plot_cumulant) and the explicit echo of every printed line is a genuinely good
  on-ramp.
- The gotcha boxes (macOS fork crash, bilateral overflow guard, +10 cap,
  edge-label identity) are precisely the sharp edges a newcomer would otherwise
  hit blind.

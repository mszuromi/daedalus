# Plan: integrate the Dyson–Duhamel expansion (unequal‑diffusion propagator)

**Status (June 2026): step‑1 FOUNDATIONS done; coupled e2e wiring is the next phase.**
This wires the paper's Appendix‑B §B.24–B.30 Dyson/Duhamel series into the spatial
pipeline so **coupled multi‑field theories with unequal diffusion** (`𝒟̂ ≠ 0`) become
computable. Today the code is hard‑gated to a single scalar diffusion `D` (`𝒟̂ = 0`),
so only the `n=0` term is realized. Builds on `docs/theory_builder_split_plan.md`.

**Done (validated, committed):**
- **Spectral reference propagator** `G₀` — `msrjd/integration/spatial/spectral_propagator.py`
  (`D₀` split, eigenprojectors `P_α`, `G₀=Σ_α P_α e^{−(m_α+D₀|k|²)t}`). Commit `24135f1`.
- **`M`/`𝒟` extraction** from the symbolic `K_ft` — `heat_kernel.reaction_diffusion_matrices`.
  Commit `0205e42`. (Full chain `K_ft → M,𝒟 → G₀ == expm` validated.)
- **(3a) spectral-Lyapunov tree 2-point** — `spectral_propagator.lyapunov_covariance` /
  `coupled_two_point` (`C(q,τ)=e^{−A|τ|}Σ`, `A Σ+Σ Aᵀ=N`). Commit `ab75084`. Validated
  vs scipy Lyapunov + a 2-species OU simulation + diagonal mode-sum reproduction.
- **(3b) coupled scalar-diffusion tree-level e2e** — `extract_noise_matrix` (matrix `N`)
  + `pipeline_bridge.compute_coupled_tree_correlator` + routing in
  `compute_spatial_correlator_via_pipeline` (additive; diagonal path untouched).
  Commits `7975f54`, `0bd934a`. A coupled multi-field EQUAL-diffusion theory now flows
  through `compute_cumulants` → `C_ij(x,τ)`; validated against `free_two_point`
  (decoupled limit) + exact `M`/`N` extraction. **Unequal diffusion still raises**
  (needs the dressing below).

**⚠ FINDING that refines this plan (June 2026).** The plan below ("feed the dressing
through the existing mode machinery") is correct for the **loop dressing** (the
per‑edge `|k|^{2n}` insertions, §3). But the *base coupled propagator and its tree
2‑point* need a genuine generalization first: the current tree 2‑point
(`pipeline_bridge._modes_C_q_tau`) is a sum of **independent diagonal OU modes**
`Σ_α κ_α/(μ_α+D_α q²)·e^{−(μ_α+D_α q²)|τ|}`, which has **no cross‑mode terms**. A coupled
theory's free 2‑point is the **matrix Lyapunov / FDT** object
`C(q,τ)=e^{−A(q)|τ|}·Σ(q)`, `A(q)=M+D₀q²`, with `Σ(q)` solving `A Σ + Σ Aᵀ = N` (noise
matrix) — it carries cross‑mode `1/(λ_α+λ_β)` weights the diagonal mode‑sum cannot
express, and its `q→x` FT is matrix‑valued. So the coupled wiring needs, in order:
**(3a) ✅ DONE** a spectral‑Lyapunov tree 2‑point (validated vs
`scipy.linalg.solve_continuous_lyapunov` + a 2‑species OU sim), **(3b) ✅ DONE** the
coupled scalar‑`𝒟` tree‑level e2e via a dedicated driver reading `prop['K_ft']`
(no gate surgery needed — `build_propagator` always keeps `K_ft`), **(3c) REMAINS**
the loop‑level matrix‑propagator integrator (projector vertices; lift
`pipeline_bridge.py:823`), THEN the §3 Dyson dressing for `𝒟̂≠0` (unequal diffusion).
The reaction‑matrix diagonalization is reusable across all of these.

## What it is, and why

For a general `N`‑field theory the reaction matrix `M = diag(μ_i) − A^(0)` and the
diffusion matrix `𝒟 = diag(D_i) + A^(2)` need not commute. The paper avoids a full
matrix heat kernel `e^{−𝒟|k|²t}` by splitting `𝒟 = D_0 I + 𝒟̂` and treating
`𝒟̂|k|²` as a **perturbative derivative insertion**:

```
G_R = (K_0 + 𝒟̂|k|²)^{-1} = Σ_{n≥0} (−1)^n G_0 [𝒟̂|k|² G_0]^n         (B24)
G_n(t,k) = (−1)^n |k|^{2n} e^{−D_0|k|²t} 𝓗_n(t)                          (B26)
𝓗_n(t)  = Σ_{α_0..α_n} P_{α_0}𝒟̂P_{α_1}⋯𝒟̂P_{α_n} e^{−m_{α_0}t}
           · Φ_n(t; m_{α_1}−m_{α_0}, …, m_{α_n}−m_{α_0})                  (B27)
```

`G_0 = Σ_α P_α e^{−(m_α+D_0|k|²)t}` is the scalar‑`D_0` reference propagator the code
already builds (per diagonal mode). `Φ_n` is the nested‑simplex / Hermite–Genocchi
divided difference of `τ↦e^{−τt}` on the eigenvalue nodes `{m_α}` (closed form even
when modes coincide — the confluent limit). In §3 this surfaces as a **per‑edge sum**
`Σ_{n_e} (−1)^{n_e}|k_e|^{2n_e} 𝓗_{n_e}(w_e)` on each retarded edge.

## Central insight: Dyson insertions are a per‑EDGE DRESSING

The expansion does **not** create new diagrams — enumeration is unchanged. Each
retarded edge `e` is *dressed* into a finite sum over `n_e` insertions, and the
diagram value factorizes into two pieces that the existing pipeline already knows
how to integrate:

1. **Momentum part `(−1)^{n_e}|k_e|^{2n_e}`** — an even polynomial in the routed
   edge momentum `k_e = Σ B_er λ_r + Σ C_eb q_b`. This is *exactly* the same object
   as a derivative‑vertex form factor (`Rcal`). **It folds into the existing per‑edge
   form‑factor product and is integrated EXACTLY by the existing Wick‑moment / GH
   loop average** — no new momentum integrator.
2. **Coefficient `𝓗_{n_e}(w_e)`** — a matrix‑valued, momentum‑INDEPENDENT time
   function. **It folds into the non‑momentum prefactor `𝒞(w)`** and rides the
   existing causal‑chamber time integral. It needs one new numerical primitive: the
   `Φ_n` divided difference.

So the whole diagram becomes
```
⟨…⟩^(Γ) = Σ_{{n_e}}  ∏_e [ (−1)^{n_e}|k_e|^{2n_e}  →  Rcal ]
                    × [ 𝓗_{n_e}(w_e)               →  𝒞(w) ]
```
where the `{n_e}` sum runs over each retarded edge's `0…N` (the truncation order).

## The one genuinely new primitive: the `Φ_n` evaluator

`Φ_n(t; ν_1,…,ν_n) = ∫_{σ_n} t^n e^{−t Σ u_i ν_i} d𝐮` = divided difference of
`e^{−τt}` on `{m_α}`. **The temporal pipeline already computes this class of
nested‑exponential simplex integral** (`final_integral._exp_over_chain_simplex`, and
the proposed Opitz‑via‑`expm`‑of‑a‑bidiagonal primitive for confluent nodes —
`docs/m_ge3_chain_simplex_fix_proposal.md`). Adapt it to `Φ_n` (this is one of the
spatial←temporal lessons in `docs/temporal_lessons_from_spatial.md`).

## Keeping `compute_cumulants` simple: order is a MODEL property

The truncation order is **set at build time and read from the model** — never a
`compute_cumulants` argument. The signature does not change.

```python
SpatialTheoryBuilder(...).diffusion_matrix(...).dyson_order(2)...build()
# → model['spatial']['dyson'] = {'mode': 'fixed', 'order': 2}
```

`compute_cumulants` / `pipeline_bridge` read `model['spatial']['dyson']` and truncate
the `{n_e}` sum accordingly. The complexity is fully encapsulated in the spatial
integration path; the temporal pipeline and the `compute_cumulants` API are untouched.
An optional env escape hatch `SPATIAL_DYSON_ORDER` overrides the model for
experimentation (mirrors the existing `SPATIAL_INTEGRATOR` pattern).

## Generality options — a POLICY field, not a bare int

To not be boxed in by the fixed‑order v1, make `model['spatial']['dyson']` a **policy
object** that the integrator dispatches on. The schema accommodates every future mode
**without changing `compute_cumulants`**:

| mode | `dyson` policy | when chosen | cost | status |
|---|---|---|---|---|
| **A. fixed** | `{'mode':'fixed','order':N}` | user picks `N` | `(N+1)^{|E_R|}`/diagram | **v1** |
| **B. auto‑tol** | `{'mode':'auto','tol':ε}` | `N` chosen at propagator build from `‖𝒟̂‖/D_0` so the next term `< ε` | same, `N` derived | v2 |
| **C. per‑edge adaptive** | `{'mode':'adaptive','tol':ε}` | truncate each edge's sum at runtime when the term drops below `ε` (high‑`|k|` edges need fewer) | ≤ A, data‑dependent | v3 |
| **D. resummed / exact** | `{'mode':'resum'}` | no truncation: sum the geometric series → dressed `k`‑dependent eigenvalues `m_α(k)` = eigenvalues of `M + 𝒟|k|²` (the full matrix heat kernel the `D_0`‑split avoids) | matrix resolvent per `(k,mode)`; loses the clean `D_0`‑factorization → a different, harder integration path | v4 |

Design rule that buys the generality: **v1 implements only `mode='fixed'`, but the
model field, the policy dispatch, and the per‑edge‑dressing machinery are shaped so
that B/C/D are added later as new branches of the same dispatch** — same model key,
same `compute_cumulants`, same enumeration. A→B is a build‑time convergence
estimator; A→C moves the truncation decision into the integrator's per‑edge loop;
A→D swaps the per‑edge dressing for a resolvent (the genuine matrix‑heat‑kernel route,
documented as the long‑term general case).

Also keep an explicit `mode='off'` (≡ `𝒟̂=0`, today's behavior) so scalar‑diffusion
theories pay nothing.

## Technical pipeline implementation

**(1) Build time — `SpatialTheoryBuilder` (new methods):**
- Accept coupled fields (lift the diagonal‑only restriction at the authoring layer).
- Parse the diffusion matrix `𝒟` from the action's Laplacian coefficients (incl. any
  cross‑field `A^(2)`). Choose the reference `D_0` (default: mean or min eigenvalue of
  `𝒟`, or a user `reference_diffusion(D0)`); set `𝒟̂ = 𝒟 − D_0 I`.
- `dyson_order(N)` / `dyson(mode=…, …)` → `model['spatial']['dyson']`.
- Validation: `𝒟̂≠0` with `dyson.mode='off'/order=0` ⇒ warn (uncontrolled error).

**(2) Propagator build — `heat_kernel.py` (the new machinery):**
- Build the reaction matrix `M`, diagonalize → `Ψ`, eigenvalues `m_α`, spectral
  projectors `P_α` (lift the off‑diagonal `K_ft` reject `heat_kernel.py:308`).
- Store `D_0`, `𝒟̂` in the eigenbasis (`𝒟̂_{αβ}=P_α𝒟̂P_β`), the mode set, and the
  `dyson` policy. Reference propagator stays the scalar‑`D_0` heat kernel.

**(3) Per‑edge dressing — `pipeline_bridge.py`:**
- For each retarded edge, generate the `n_e=0…N` terms: deposit `(−1)^{n_e}|k_e|^{2n_e}`
  into the per‑edge `Rcal` factor (extend `diagram_form_factor`), and `𝓗_{n_e}(w_e)`
  into `𝒞(w)` (extend the non‑momentum prefactor; call the `Φ_n` evaluator).
- The `{n_e}` product‑sum over edges → `∏_e (Σ_{n_e})`. Each assignment runs the SAME
  integrator.

**(4) Integration — `full_integrator.py` (mostly unchanged):**
- The `|k_e|^{2n_e}` insertions are absorbed by the existing Wick‑moment / GH average
  (they are even polynomials in the loop/external momenta). The `𝓗_{n_e}(w_e)`
  amplitude multiplies the per‑chamber `𝒞(w)` weight. No new integrator.

**(5) Truncation / cost:** finite `(N+1)^{|E_R|}` assignments per diagram (`|E_R|` =
retarded edges). `N=1, |E_R|=3` → 8×; `N=2` → 27×. Fine for small `N`; the per‑edge
adaptive policy (C) prunes this later.

## Phasing

1. **Spectral propagator** (coupled `M`: `Ψ, m_α, P_α`; `D_0` split; `𝒟̂`) behind
   `SpatialTheoryBuilder` — validate the `n=0` term reproduces today's diagonal result.
2. **`Φ_n` evaluator** (adapt the temporal chain‑simplex / Opitz primitive) +
   unit‑test vs a brute‑force nested‑simplex quadrature and the confluent limit.
3. **Per‑edge dressing** + the `{n_e}` sum (`mode='fixed'`), wired through the existing
   form‑factor + `𝒞(w)` machinery. Validate a 2‑field unequal‑diffusion theory at
   `N=1,2` vs a direct matrix‑heat‑kernel oracle (brute `∫dℓ` with `e^{−𝒟|k|²t}`).
4. **Policy dispatch** (`off/fixed`) + env escape hatch; then v2 `auto‑tol`.

## 3c plan: loop-level coupled integrator via SPECTRAL ASSIGNMENTS (June 2026)

**Scope:** loop corrections (`max_ell≥1`) for coupled fields with **scalar diffusion**
(`𝒟̂=0`) — the loop companion of 3a/3b. Unequal diffusion stays gated until the
dressing below.

**Central reduction.** With scalar diffusion every edge's momentum factor is the SAME
heat kernel `e^{−D₀k²w}` as today — `D₀` is field-independent, so the entire
Symanzik/heat-kernel-IFT/Wick/Bessel machinery is untouched. The coupling lives ONLY
in the time/matrix factor: each retarded edge carries `G₀(w_e,k)=Σ_α P_α e^{−m_α w_e}
· e^{−D₀k²w_e}`. Expanding every edge in its spectral components turns one coupled
diagram into a finite sum of SCALAR diagrams:

```
value(Γ) = Σ_{{α_e}}  W({α_e}) × I_scalar({m_{α_e}}; D₀, …)
W({α_e}) = full field-index contraction of [P_{α_e}] over the diagram's vertex
           coupling tensors, the noise matrices N at source vertices, and the
           external legs   (a pure numeric tensor network — no integrals)
I_scalar  = the EXISTING chamber integral, with per-edge masses m_{α_e}
```

Consistency anchor: at tree level this machinery must reproduce 3a exactly —
`Σ_{αβ} P_α N P_βᵀ/(m_α+m_β)` IS the spectral form of the Lyapunov solution.

**The one real integrator change — per-edge masses.** `diagram_kinematic` currently
hoists a UNIFORM `mu` out of the chamber integral (`exp(−mu·s_nodes)` at
`full_integrator.py:474`; the analytic `mu**(n_V+n_C)` marginal at :566). 3c
generalizes `C(w)`'s mass factor to `exp(−Σ_e m_e w_e)` with per-edge `m_e` (complex
allowed — conjugate eigenvalue pairs; the assignment sum is real). The uniform-mass
analytic shortcuts must be re-derived per-edge or bypassed for the coupled path
(diagonal theories keep the uniform fast path bit-identically).

**Steps (each test-gated):**
1. **3c-1 per-edge masses in `diagram_kinematic`** (+ `diagram_correlator/_x`):
   `mu: float` → `mu: float | ndarray[n_edges]`; uniform input must stay
   bit-identical (regression-pinned). Validate per-edge vs brute ∫dℓ oracle on the
   bubble with two different masses.
2. **3c-2 spectral-assignment driver** in `pipeline_bridge`: read `M,D₀,N` from the
   3b info dict → `m_α,P_α`; enumerate `{α_e}` per diagram; build `W({α_e})` by
   contracting projectors at vertices (per-node vertex-type table from the lowering,
   task #143 machinery) + `N` at noise sources; call 3c-1 per assignment; sum.
3. **3c-3 lift the single-mode gate** (`pipeline_bridge.py:946` block): dispatch
   `len(modes)==1` → today's path (bit-identical), coupled scalar-D → 3c-2.
4. **3c-4 validation:** (i) tree-level via {α_e} == 3a Lyapunov (anchor); (ii) a
   decoupled 2-field theory == two independent single-field runs (exactness);
   (iii) coupled 2-species RD with a nonlinearity, `max_ell=1`, vs the 2-species
   spatial Langevin simulator (physics oracle); (iv) full diagonal+loop regression.

**Cost:** `(N_fields)^{|E_R|}` scalar integrals per diagram (2-field bubble ≈ 8) —
same shape as the Dyson `{n_e}` sum; the two compose multiplicatively later, which is
fine at small N.

## Dyson dressing plan (unequal diffusion, 𝒟̂≠0) — after 3c

1. **D-1 `Φ_n` evaluator** (`spectral_propagator.py`): divided difference of
   `e^{−τ t}` on eigenvalue nodes via the Opitz/`expm`-of-bidiagonal form (nodes on
   the diagonal, `t` on the superdiagonal) — confluent-safe by construction. Unit-test
   vs brute nested-simplex quadrature + the equal-nodes limit `t^n e^{−mt}/n!`.
2. **D-2 `𝓗_n(w)` assembly:** projector strings `P_{α_0}𝒟̂P_{α_1}⋯𝒟̂P_{α_n}`
   × `e^{−m_{α_0}w}` × `Φ_n(w; m_{α_i}−m_{α_0})` (B27). This EXTENDS 3c-2's
   assignment machinery: an edge's label grows from `α_e` to
   `(n_e, α_0..α_{n_e})`; `W` picks up the 𝒟̂ string factors.
3. **D-3 momentum side:** the `(−1)^{n_e}|k_e|^{2n_e}` insertion folds into the
   per-edge form factor `Rcal` (even polynomial in the routed momentum → existing
   GH/Wick-moment average, exact). Time side: `𝓗_{n_e}(w_e)` multiplies `𝒞(w)`
   via the (already per-edge after 3c-1) mass factor.
4. **D-4 builder + policy:** `SpatialTheoryBuilder.dyson_order(N)` /
   `.reference_diffusion(D0)` → `model['spatial']['dyson']={'mode':'fixed','order':N}`
   (+ `'off'`); build-time `‖𝒟̂‖/D₀` convergence warning; `SPATIAL_DYSON_ORDER` env
   escape hatch. `compute_cumulants` signature unchanged.
5. **D-5 validation ladder:** tree-level unequal-D 2-point at `N=0,1,2` vs the exact
   `expm(−(M+𝒟q²)τ)`+Lyapunov oracle (convergence rate ~ `(‖𝒟̂‖/D₀)^{N+1}`);
   then loop-level vs brute ∫dℓ with the matrix heat kernel; then a 2-species
   unequal-D Langevin sim.

## Risks
- **Convergence**: the series needs `‖𝒟̂‖/D_0` small; a bad `D_0` choice diverges.
  The auto‑tol policy (B) and a build‑time convergence check mitigate this.
- **Cost blow‑up** at high `N`/many edges — bounded by keeping `N` small + the
  adaptive policy (C).
- **`Φ_n` at confluent modes** (coincident `m_α`) — handled by the Opitz/`expm`
  divided‑difference (no `0/0`).
- **Scope creep into compute_cumulants** — explicitly avoided: policy in the model,
  dressing internal to `pipeline_bridge`, signature unchanged.

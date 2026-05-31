# Backend C — mathematical foundation (real-time Symanzik + sector decomposition)

**What this is.** The math that backend C rests on: the parametric (Schwinger)
representation of MSR-JD loop integrals, the Symanzik polynomials in our
heat-kernel time representation, the residual *causal* time-simplex integral, its
singularity structure, and how sector decomposition + dimensional regularization
turn it into a finite, renormalized result at **arbitrary loop order L and
spatial dimension d**.

**Companion docs.** Engineering plan & milestones: `docs/backend_C_design.md`.
High-level decision (backend A now, C later) and reading list:
`docs/spatial_v2_architecture.md` §5 (option C), §9. This note is the detail §9
calls "the research content of the eventual C backend."

**One-line claim.** In the heat-kernel `(k,t)` representation the *edge times are
already Schwinger parameters*, so the loop momentum integral is Gaussian at any L
and any d — it collapses to the Symanzik polynomials `U(w), F(w)` — and the only
genuinely new work is the *causal* time-parameter integral that remains.

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
is to do it after the Symanzik reduction (smooth in momentum) and to handle its
boundary singularities systematically (§4–5).**

---

## 4. Singularity structure

Two singularity types, both on the boundary of the `w`-domain:

**(a) UV / small-time.** `U(w)` is homogeneous of degree `L`, so as all `w_e→0`
together `U→0` and `U^{−d/2}→∞`. The local integral behaves like
`∫ w^{(#edges)−1−Ld/2} dw`, i.e. it diverges when `Ld/2 ≥ #edges − L` — the UV
divergence, and it **worsens with d and L**. This is exactly the
`σ_R(a)~a^{−1/2}` integrable singularity hand-fixed this session with the
power-law sliver in `loop_dyson.bubble_delta_C_q_tau` (`F_R=q²ℓ²` ⇒ `U^{−1/2}` at
small time). The sliver *is* a 1-loop, one-variable, by-hand instance of what
sector decomposition does in general.

**(b) Close-pair / near-degenerate.** When two edge masses `m_{k_e}, m_{k_{e'}}`
nearly coincide, the analytic pole-residue form of the time integral carries
`(e^{−mt}−e^{−m't})/(m−m')`, which loses precision as `m→m'` — the close-pair
bug. In the parametric representation this is a near-coincidence of parameters: a
structured, milder boundary behaviour that sector decomposition's "split and
remap" also tames. **This is why C is the principled cure for close-pair, not a
per-diagram patch.**

---

## 5. Sector decomposition + dimensional regularization

Sector decomposition (Hepp; Binoth–Heinrich; implemented in pySecDec,
arXiv:2202.13647) makes the `w`-integral finite and numerically tractable:

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

**Covers:** the loop evaluation at arbitrary `L`, `d`, with systematic UV
renormalization and a principled close-pair cure. The momentum step (Symanzik) is
mature and `d`-general; the open research piece is the *causal* time-simplex
integral + sector decomposition adapted to retarded/Keldysh propagators (§3–5).

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

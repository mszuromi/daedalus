# Conductance-style interaction vertices: kernel factors at non-(1,1) sectors

**Status:** Design — implementation in progress
**Branch:** `convolution-operator`
**Date:** 2026-05-17

## Problem statement

When the action contains a conductance-style term

```
S_int ∋ ∫ dt  vt(t) · v(t) · Conv(g, n)(t) · κ
```

(kernel `g` convolved with the field `n`, multiplied by the *other* field
`v` outside the convolution), the saddle expansion `v = vstar + v',
n = nstar + n'` produces — among others — an **interaction vertex** at
bigrade `(1, 2)`:

```
S_(1,2) ∋  κ · g · vt · v' · n'
```

The kernel SR symbol `g` ends up in the **vertex coefficient**, not just
the (1,1) bilinear. The existing pipeline substitutes `g → ĝ(ω)` only in
the (1,1) matrix `K_ft` ([pipeline/_propagator.py:563-566](../pipeline/_propagator.py)) —
interaction-vertex kernel factors are left as raw SR symbols and would
either:

- survive into the numerical integrator and produce a JIT-compilation
  failure when `fast_callable` tries to evaluate them numerically, or
- silently get treated as a free parameter and produce wrong numbers.

The existing infrastructure for non-local **source** vertices
(`NoiseSourceType` at [msrjd/core/vertices.py:110](../msrjd/core/vertices.py)) is the right shape but
covers only auto-cumulant noise sources (`correlated_noises` block), not
interaction vertices with conductance-multiplicative synapses.

## Proper math

### Time-domain form

A vertex term `κ · vt(t) · v(t) · Conv(g, n)(t)` is

```
∫ dt vt(t) v(t) ∫ ds  g(t − s) n(s)  · κ
= ∫∫ dt ds  vt(t) v(t) n(s)  ·  g(t − s)  ·  κ
```

The vertex is **non-local in time**: the `n`-leg sits at time `s` ≠ `t`,
bridged by the kernel `g(t − s)`. This is exactly the structure
`NoiseSourceType` already handles for cumulant kernels — the integrator
treats response legs of a noise source as sitting at independent times
linked by the cumulant kernel `κ(τ)`.

### Fourier-domain form

By the convolution theorem,

```
∫ dt e^{−iωt} (g ⋆ n)(t)  =  ĝ(ω) · n̂(ω)
```

so for the (1, 2) vertex above, the diagrammatic weight in
momentum space is

```
V^{(1,2)}(ω_vt, ω_v', ω_n')  =  κ · ĝ(ω_n')
```

with overall momentum conservation `ω_vt + ω_v' + ω_n' = 0`. The
kernel factor `ĝ(ω)` is **evaluated at the frequency of the leg the
kernel was attached to** — here, the `n'` leg.

### Key invariant

The kernel symbol `g` in a vertex coefficient is attached to **exactly
one leg** — the field that appeared as argument 2 of the original
`Conv(g, ·)` in the action. This is preserved by `reduce_conv_in_action`'s
rule 4 `Conv(g, fluct) → g · fluct`: the product is *adjacent* in SR
before `.expand()` runs. But after bigrade classification the coefficient
sits in the polynomial-ring monomial coefficient bucket and the
adjacency is lost.

We need to **record the (kernel_symbol, attached_field) association
during Conv reduction** so vertex extraction can recover it.

## Implementation plan

### Phase 1 — record kernel attachments (this commit)

Extend `reduce_conv_in_action` with an optional out-parameter
`attachments_out: dict` that the reducer writes to. Keyed on the
kernel SR symbol, value is the set of fluctuation fields the kernel
was paired with by rule 4. For the single-Conv case it has one entry;
for theories with several Conv's it accumulates.

```python
reduce_conv_in_action(expr, fluct_vars, taylor_order=4,
                      attachments_out=attachments)
# After: attachments[g_symbol] = {n_symbol}
```

Add a detection helper

```python
def kernel_attachment_for_coefficient(coeff, attachments, leg_vars):
    """For each kernel symbol present in coeff, return the leg
    variable it was attached to (looked up from ``attachments``).
    Returns dict {kernel_symbol: leg_var}."""
```

Tests:
- Linear `Conv(g, n)`: attachments = {g: {n}}.
- Conductance `v · Conv(g, n)`: same attachments.
- Multiple `Conv(g1, n1) + Conv(g2, n2)`: attachments correctly
  splits {g1: {n1}, g2: {n2}}.
- Same kernel reused `Conv(g, n1) + Conv(g, n2)`: attachments[g] =
  {n1, n2} — caller must disambiguate per-vertex by index matching.

### Phase 2 — propagate to vertex extraction (next session)

Modify `extract_vertex_types` ([msrjd/core/vertices.py:252](../msrjd/core/vertices.py)) to:

1. After decomposing a sector into monomials, for each monomial scan
   its coefficient for kernel symbols listed in
   `model['kernels']`.
2. Use `attachments` (stashed on `ns` during expand) to determine
   which leg each kernel symbol couples to.
3. Either:
   - Wrap the monomial in a new `ConvVertexType` subclass that
     carries `kernel_attachments: [{'symbol': g, 'leg': (n, j),
     'ft_image_fn': lambda ω: ĝ(ω)}]`, or
   - Strip the kernel symbol from the coefficient and attach the
     kernel-leg pair as metadata.

### Phase 3 — wire into the integrator (next session)

For Phase J time-domain integration: each `ConvVertexType` adds an
effective convolution on its kernel-attached leg's propagator. In
Fourier-image form this is simply `G_leg(ω) → ĝ(ω) · G_leg(ω)`,
equivalent to convolving the time-domain propagator with `g(t)` once
before composing.

For the equivalent frequency-domain integrator
(`msrjd/integration/symbolic.py`), the modification is direct: insert
the `ĝ(ω_leg)` factor at the vertex like the existing
`NoiseSourceType` path does at lines 364-366.

## Compatibility

- Theories with no `Conv` in the action: untouched. The `attachments`
  dict stays empty, no vertex extraction changes fire.
- Theories with `Conv(g, n_linear)` where the kernel only appears in
  the (1,1) bilinear (like the current quad-exp theory): also
  untouched. The existing propagator FT path handles these.
- Conductance-style theories with `v · Conv(g, n)` or
  `phi(v) · Conv(g, n)`: Phase 1 collects the attachments; Phases 2–3
  consume them.

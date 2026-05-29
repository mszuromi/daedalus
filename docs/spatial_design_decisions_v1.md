# Spatial v1 — design decisions

**Date**: 2026-05-28
**Status**: Locked for v1 implementation; revisit at v1.5 / v2 boundaries.
**Companions**:

- `docs/spatial_implementation_outline.md` — v1 design rationale (the *what* and *why*)
- `docs/spatial_implementation_plan.md` — per-phase code-change plan (the *how*)

This document is Phase 0's deliverable: the four open design questions
that the plan flagged, each with a chosen option and implementation
notes for any place the choice diverges from the plan's default
recommendation.

---

## Decision 1 — `spatial_dim` declaration model

**Chosen**: **per-field data model with a "set all fields to the same
dim" convenience surface**.

Underneath the hood, every `FieldSpec` carries its own `spatial_dim`
(int, default 0).  The user-facing API has two entry points:

1. **Primary**: `physical_field(name, spatial_dim=d, ...)` — per-field
   declaration.  This is what the framework actually consumes; every
   other surface ultimately calls it.
2. **Convenience**: `.spatial_dim(d)` builder method — sets `spatial_dim`
   on **all** fields currently declared on the builder, AND establishes
   `d` as the default for any field declared afterwards.  Per-field
   explicit settings (e.g., `physical_field('m', spatial_dim=0)` after
   `.spatial_dim(1)`) override the default.

The Theory Builder UI (`pipeline/ui/`) will expose this as a single
**"All fields same spatial dim"** toggle on the Fields tab.  When
ticked, the UI shows one numeric input for `d` and bulk-applies via
`.spatial_dim(d)`.  When unticked, each field row gets its own
`spatial_dim` input.

### v1 validation

At `build()`, the framework asserts that
`len({f.spatial_dim for f in fields if f.spatial_dim > 0}) <= 1` —
i.e., all fields with non-zero `spatial_dim` agree on the dim
value.  Mixing dim=1 and dim=2 fields is **out of v1 scope** and
raises `NotImplementedError`.  Mixing dim=0 (a spatially-averaged
auxiliary field, e.g., a time-only `m(t)`) with dim≥1 spatial
fields is **allowed in v1** and is the reason the data model is
per-field rather than per-theory.

### Why this rather than per-theory `.spatial_dim(1)` only

The per-theory option closes the door on spatially-averaged
auxiliary fields (think: `φ(x, t)` coupled to a global order
parameter `m(t)`).  The per-field model gets that case "for free"
in v1 and gets full mixed-dim support "for free" in v2.  The
convenience method preserves the per-theory feel for the common
single-dim case.

### Implementation impact on the plan

- **Phase 1.1** gains a `.spatial_dim(d)` builder method
  (~30 LOC) in addition to the per-field `physical_field(spatial_dim=)`
  kwarg.  Method body: walks `self._fields` and sets each
  `FieldSpec.spatial_dim = d`, and sets `self._default_spatial_dim = d`
  for fields declared after the call.
- **Phase 1.4 docstring** for `.boundary(...)` should reference that
  it only makes sense after at least one field has `spatial_dim > 0`;
  raise a clear error if invoked on a time-only builder.
- **Phase 6 UI extension** (~80 LOC in `pipeline/ui/widgets.py`):
  - Add "All fields same spatial dim" checkbox to the Fields tab
  - When checked: one numeric input shown; per-row inputs hidden
  - When unchecked: per-row `spatial_dim` numeric inputs shown next
    to the existing field controls
  - JSON round-trip preserves the checkbox state alongside the
    per-field values
- Validation message: `"v1 does not support mixed spatial dimensions:
  field 'phi' has dim=1, field 'psi' has dim=2.  See
  docs/spatial_implementation_plan.md §'Out of v1 scope' for the v2
  mixed-dim plan."`

---

## Decision 2 — PBC boundary length

**Chosen**: **named parameter** (with inline-number as a syntactic
shortcut).

`.boundary('periodic', length='L')` references a separately-declared
`.parameter('L', default=20.0, domain='positive')`.  `L` is sweepable
like any other parameter; the PBC→∞ checkpoint test becomes a one-
parameter sweep.

### Inline-number shortcut

For theories that never sweep `L`, the syntactic shortcut
`.boundary('periodic', length=20.0)` is accepted: the framework
auto-creates a hidden parameter `_pbc_length_L0` with default
`20.0` under the hood, and the user can ignore the indirection
unless they want to sweep.

### Validation

If `length` is a string, framework looks up the named parameter and
raises `KeyError` with a clear message if not found.  If numeric, no
lookup; just use the value.  The string form is the recommended
default in documentation and the test theory; the numeric form is
documented as a shortcut for quick prototyping.

### Implementation impact

- **Phase 1.4** validation in `.boundary(...)`: accept either type;
  if string, defer lookup to `build()` so order of `.parameter()` /
  `.boundary()` calls doesn't matter.
- **Phase 2.x** propagator builder: receives `bc_params['length']`
  resolved to a numeric value (the parameter's default, since
  propagator construction happens at build-time).
- **Phase 6 test theory** `allen_cahn_1d_subcritical_pbc.theory.py`
  uses the named-parameter form to exercise the more general
  pathway.

---

## Decision 3 — `max_ell > 1` on spatial theory (rescue tier None or A)

**Chosen**: **soft warning + scipy.quad fallback**.

When `model.get('spatial')` exists AND `max_ell > 1` AND the chosen
rescue tier is None or A (see Decision 4), `compute_cumulants` emits
a `UserWarning` and routes m≥2 subsets through `scipy.quad`.

### Warning message (exact text)

```
UserWarning: max_ell=2 on a spatial theory uses the slow scipy.quad
fallback at m≥2 subsets (no closed-form path available at this
rescue tier).  Wall time may exceed 10× the m=1 case.  For closed-
form 2-loop, see docs/spatial_implementation_plan.md §"Heat-kernel
rescue paths" (Rescue B).
```

Emitted **once per `compute_cumulants` call**, not per subset.
Catchable via `warnings.catch_warnings(...)` in tests so the
regression suite doesn't churn on warning text.

### Implementation impact

- **Phase 5.5** lands as soft-warning + `scipy.quad` fallback path
  unchanged from the plan's Variant 5.4-None description (since A
  doesn't implement m≥2 closed-form, the m≥2 fallback is still
  scipy.quad even under Rescue A).
- The warning machinery uses `warnings.warn(...,
  stacklevel=2)` so the user's calling line is the one flagged.

---

## Decision 4 — Heat-kernel rescue tier

**Chosen**: **Rescue A** — Path 1 m=1 erfc closed form replaces
`scipy.quad` at m=1.

### What this commits v1 to

- **Phase 5 Variant 5.4-A** is the active path; Variant 5.4-None is
  abandoned (no scipy.quad fallback at m=1 in v1).
- The m=1 closed-form helper `_integrate_1d_polytope_with_erfc` is
  built such that Phase 5b's m=2 / m≥3 extensions (Rescue B) can
  reuse it as the leaf of higher-m recursions.  This shapes the
  function's API and unit-test coverage.
- `mpmath` safety-net flag `USE_ERFC_CHAIN_SIMPLEX_PRECISION_FIX`
  is added as flag-gated dead code from day one, default `False`.
  Parallels `USE_CHAIN_SIMPLEX_PRECISION_FIX` from the time-domain
  chain-simplex audit.

### What this defers to follow-on

- **Phase 5b (Rescue B's m=2 + m≥3 erfc extensions)**: not in v1.
  When `max_ell > 1`, the soft-warning + scipy.quad path from
  Decision 3 fires.  Rescue B becomes available as a separate
  branch off `spatial-extension` if/when d=1 2-loop becomes a
  near-term priority.
- **Rescue C add-on** (Path 2 PBC mode-decomp cross-validation):
  not in v1.  Useful if Rescue A produces values that look off and
  we want an independent route to cross-check, but the v1
  validation suite's PBC→∞-limit test already provides one
  independent cross-check (against the lattice-sum closed form).

### Why Rescue A rather than None

Same wall-time (the erfc derivation takes ~1 wk but **replaces** the
scipy.quad fallback work that None would have spent ~1 wk on, not
**adds to** it).  Buys: 7 orders of magnitude on precision floor
(1e-8 → 1e-15), kills scipy dependency in the v1 loop integrand,
produces the m=1 erfc building block for Rescue B.

### Implementation impact

- **Phase 5.4-A** sub-section of the plan is the active spec.
- **Phase 5.5** soft-warning fires for `max_ell > 1` (Decision 3),
  routing m≥2 to scipy.quad fallback as the *separate* slow path.
- v1 total effort: **9 weeks** (unchanged from the plan's None /
  Rescue A column).

### Math spike — Rescue A premise verified (2026-05-28)

Before committing to Rescue A, a Phase 0 de-risk spike
(`docs/spatial_spikes/phase5_erfsplit_semigroup_spike.py`)
numerically verified the two load-bearing claims:

1. **The erf-split closed form** for the residual integral
   `∫_L^U s^{-1/2} exp(-β/s - α s) ds`:
   - real α, 7 regimes (small/large μ, small/large X, finite +
     near-origin-L intervals): rel ≤ 6e-15
   - **complex α** (the multi-pole-chain generality case — damped-
     oscillatory and near-pure-imaginary α=0.05+2j stress): rel
     ~1e-25 to 1e-31.  **This is the result that confirms Rescue A
     generalizes past free Allen-Cahn** — the erf-split has no
     hidden instability in the oscillatory-pole regime.
   - semi-infinite U→∞ (the τ-correlator integral): rel ≤ 2e-16
2. **The spatial semigroup collapse** `∫dx_v G(t1,·)G(t2,·) =
   G(t1+t2,·)`, chain-2 and chain-3: rel ≤ 3e-16.
3. **End-to-end tree-level free correlator** C(x,τ): 3-way match
   (erf-split closed form vs direct 2D quadrature vs analytic);
   at τ=0 reproduces `T/(2√(μD))·exp(-|x|/ξ)` exactly.

**Verdict: GO.**  The spike uses mpmath at 30 dps, so it confirms
the math is correct; the float64 production path still needs the
`USE_ERFC_CHAIN_SIMPLEX_PRECISION_FIX` mpmath gate for the
close-erf-argument regime (already specced — same character as the
chain-simplex close-pole concern in
`docs/m_ge3_precision_bug_audit.md`).

---

## Summary of v1 commitments

| Decision | Choice | Net plan impact |
|---|---|---|
| D1: spatial_dim model | Per-field data model + `.spatial_dim(d)` convenience method + UI "all-same" toggle | +~30 LOC builder method; +~80 LOC UI |
| D2: PBC length | Named parameter (string), inline-number shortcut accepted | No effort change |
| D3: max_ell > 1 behaviour | Soft warning + scipy.quad fallback at m≥2 | No effort change |
| D4: Rescue tier | Rescue A (Path 1 m=1 erfc) | No effort change; Phase 5b deferred |

**v1 total effort: ~9 weeks** (matches the plan's Rescue A column).

---

## Out-of-v1 scope clarifications (post-decision)

These were already listed in the plan's "What is explicitly out of v1
scope" section; reaffirming here so the decisions don't drift:

1. **Mixed spatial dimensions** within a single theory (dim=1 + dim=2
   fields) — defer to v2.  Per-field data model from D1 enables this
   for free when v2 wants it.
2. **Phase 5b (Rescue B's m=2 + m≥3 erfc)** — defer.  The hard wall
   at `max_ell > 1` is the soft-warning behaviour from D3 with the
   m≥2 scipy.quad fallback; if a user runs into wall-time pain on
   2-loop, that's the trigger for Rescue B as a follow-on.
3. **Higher-derivative kinetic operators** (`∇⁴`, fractional `(-Δ)^α`,
   anisotropic) — defer.  These break the heat-kernel closed form
   that D4's Rescue A depends on.
4. **Dirichlet / Neumann / Robin BCs** — defer.  Only infinite and
   periodic in v1.
5. **Transient ICs** — defer.  Only stationary in v1.
6. **d ≥ 2** — defer.  v1 ships d=1 only; Path 1 m=1 erfc
   generalization to d ≥ 2 is straightforward (K_{1/2} → K_{d/2}),
   but the m=2 / m≥3 generalizations get messy.

---

## Sign-off

- User confirmed Q1-Q4 selections on 2026-05-28 via interactive
  decision prompt.
- Q1 explicitly refined from "per-field (recommended)" to "per-field
  data model with UI-side bulk-set affordance"; this document
  captures the refinement.
- Phase 0 is **complete**; Phase 1 (theory namespace + spatial
  declarations) is unblocked.

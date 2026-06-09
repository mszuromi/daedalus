# Plan: split `TheoryBuilder` → `TemporalTheoryBuilder` + `SpatialTheoryBuilder`

**Status: planned, not executed (June 2026).** Prepares the authoring layer for the
new (coupled / spectral) spatial propagator machinery (the Dyson–Duhamel work —
see `docs/dyson_duhamel_integration_plan.md`). The pipeline keeps auto‑routing on
theory type exactly as it does today.

## Governing invariant (why this is low‑risk)

The split lives **entirely at the authoring layer**. The built `model` dict schema
and every auto‑routing decision stay **byte‑identical**. Auto‑routing keys off one
emitted marker — `model['spatial']` (plus the implicit `Laplacian` operator) —
produced by one contiguous block in `build()` (`pipeline/theory.py:1620–1810`).

> **Both builders emit the same dimension‑agnostic model. Only `SpatialTheoryBuilder`
> emits the spatial marker triplet** (`model['spatial']`, `model['boundary']`,
> `model['initial']`) **+ the `Laplacian` operator + reserved‑name widening.**
> `TemporalTheoryBuilder` emits *none* of these.

If that holds, `compute.py:382` (spatial short‑circuit), `_propagator.py:789`
(heat‑kernel block), `report.py:395`, and the `theory_compiler` saddle‑killer
(`:914/:1386`) all keep working **untouched**.

## The spatial marker(s) (what build() emits, from the audit)

- **Source of truth:** `FieldSpec.spatial_dim` (per field), set via
  `physical_field(..., spatial_dim=)` or `spatial_dim(d)`.
- **Decision:** `is_spatial = any field spatial_dim ≥ 1` (`theory.py:1626–1638`).
- **PRIMARY emitted marker:** `model['spatial'] = {'dim', 'fields_with_spatial'}`,
  emitted **only when spatial** (`theory.py:1793–1800`) — the routing boolean.
- **Companions:** `model['boundary']`, `model['initial']` (`:1803–1805`).
- **IMPLICIT marker:** `Laplacian` entry appended to `model['operators']`
  (`:1701–1707`) → drives `hasattr(ns,'Laplacian')` in the compiler + saddle‑killer.
- Serializer records the *inputs* (`spatial_dim`, `boundary`, `initial`), never the
  derived `model['spatial']` (recomputed by `build()`).

## Auto‑routing decision points (stay UNCHANGED)

| site | switches on | action |
|---|---|---|
| `_propagator.py:789` | `model.get('spatial')` | add heat‑kernel block on top of temporal propagator |
| `_propagator.py:558` | `model.get('spatial')` & `G_tx_sym is None` | cache‑staleness rebuild |
| `compute.py:382` | `model.get('spatial')` & spatial_grid | **short‑circuit to `pipeline_bridge`** |
| `compute.py:376,540` | `model.get('spatial')` | spatial_grid guards |
| `report.py:395` | `config['spatial']` | no‑diagrams report branch |
| `theory_compiler.py:487,826,914,1386` | `hasattr(ns,'Laplacian')` | expose op + saddle‑killer (2 places) |
| `heat_kernel.py:275` | `model.get('spatial')` | build G_tx |

## Class hierarchy (base + two disjoint mixins + back‑compat shim)

```python
class _BaseTheoryBuilder:
    # SHARED: parameter, response_field, physical_field (spatial_dim kwarg default 0),
    #   set_action_text, equation, stability_analysis, operator_ir, define_function,
    #   set_mf_equation, all set_* setters, add_function/operator, build().
    # build(): the ~90% dimension-agnostic pipeline, then two overridable hooks:
    def _emit_domain_blocks(self, model): pass      # default: emit NOTHING
    def _reserved_extra(self): return set()

class _TemporalMethods:   # mixin
    # population(size), kernel, define_kernel, markovianize, declare_cgf_term,
    # correlated_noise, use_action_template, use_synaptic_kernel, add_gtas_noise,
    # set_transfer_function, set_kernel_ft_image

class _SpatialMethods:    # mixin
    # spatial_dim(d), boundary, initial   [+ future: diffusion_matrix, dyson_order, ...]
    def _emit_domain_blocks(self, model):  # model['spatial']+boundary+initial+Laplacian op+PBC len
    def _reserved_extra(self): return {'k','Laplacian','x','y','z'}

class TemporalTheoryBuilder(_TemporalMethods, _BaseTheoryBuilder): ...
class SpatialTheoryBuilder (_SpatialMethods,  _BaseTheoryBuilder): ...

class TheoryBuilder(_SpatialMethods, _TemporalMethods, _BaseTheoryBuilder):
    # BACK-COMPAT shim: both method sets; auto-detect the hook (current behavior)
    def _emit_domain_blocks(self, model):
        if self._is_spatial(): _SpatialMethods._emit_domain_blocks(self, model)
    def _reserved_extra(self):
        return _SpatialMethods._reserved_extra(self) if self._is_spatial() else set()
```

Mixins (not one fat class) because the two method sets are nearly disjoint and
interact only through the single `build()` hook. Forward builders give type‑specific
APIs — `.markovianize()` on a `SpatialTheoryBuilder` is a clean `AttributeError`.

## Method partition

| `_BaseTheoryBuilder` (shared) | `TemporalTheoryBuilder` | `SpatialTheoryBuilder` |
|---|---|---|
| parameter, response_field, physical_field, set_action_text, equation, stability_analysis, operator_ir, define_function, set_mf_equation, all set_* setters, add_function/operator, **build()** | population(size), kernel, define_kernel, markovianize, declare_cgf_term, correlated_noise, use_action_template, use_synaptic_kernel, add_gtas_noise, set_transfer_function, set_kernel_ft_image | spatial_dim, boundary, initial; owns `spatial_dim=` kwarg semantics; *(future: diffusion_matrix, reaction_coupling, reference_diffusion, dyson_order)* |

Subtleties: keep `spatial_dim=` kwarg on base `physical_field` (default 0);
`TemporalTheoryBuilder` validates `spatial_dim==0`. `SpatialTheoryBuilder` defaults
`n_populations=0`; multi‑field spatial uses repeated `physical_field` (not
`population`) — the seam the coupled‑propagator work needs.

## `build()` refactor (the only spatial‑branching method)

Extract the contiguous `is_spatial` region (`theory.py:1620–1810`) into
`_emit_domain_blocks(model)` (`model['spatial']` `:1793–1800`, boundary/initial
`:1803–1805`, `Laplacian` op `:1701–1707`, inline‑PBC `_pbc_length_L0` `:1724–1730`)
and `_reserved_extra()` (`:1656–1660`). Pure code motion; everything else stays in base.

## Serialization (`theory_serialize.py`) — two sub‑phases

Method‑replay (`.method(...)` chains into `TheoryBuilder(...)`, AST‑parsed back).
- **Phase‑0 (no change):** keep emitting `TheoryBuilder(...)` → all 33 `.theory.py`
  round‑trip through the shim.
- **Phase‑1 (optional):** `render_theory_file` chooses the constructor name from the
  spec (any `spatial_dim≥1`/`boundary`/`initial` ⇒ `SpatialTheoryBuilder`); AST loader
  (`:614`) recognizes all three names. Re‑render the 9 spatial + 24 temporal files.

## UI — no change required

Decoupled (spec‑dict → `render_theory_file` → text → `exec`). Split is invisible;
Phase‑1 subclass names are picked up automatically by the exec path.

## Caller blast radius (from the audit)

33 `.theory.py` (9 spatial / 24 temporal), 11 spatial notebooks, ~9 test files,
5 `pipeline/theories`, UI decoupled. **Every site uses literal `TheoryBuilder(`** →
the shim keeps all of them working with zero edits.

## Phased execution + risk

1. Refactor into base+mixins+shim; `TheoryBuilder` behavior identical. Verify: full
   suite bit‑identical + a **build‑output diff test** (model dict unchanged
   pre/post‑refactor for a temporal and a spatial theory).
2. Add forward builders + type guards; unit tests (Spatial lacks `.markovianize` &
   emits the marker triplet; Temporal lacks `.boundary` & emits no `model['spatial']`;
   subclass model dicts == shim's).
3. *(optional)* serializer Phase‑1 + re‑render + migrate spatial notebooks/tests.

**Main risk:** the shim's auto‑detect must reproduce current behavior exactly —
caught by the build‑output diff test.

## Open decisions
1. Keep `TheoryBuilder` shim (recommended) vs hard‑migrate ~50 sites now.
2. Serializer Phase‑0 only (recommended) vs Phase‑1 now.
3. base+2 mixins+shim (recommended) vs base+2 subclasses + thin alias.
4. First PR scope: refactor + working forward builders only (recommended) vs also stub
   new spatial‑propagator authoring methods.

**Recommendation:** #1 shim, #2 Phase‑0, #3 base+mixins, #4 refactor + forward builders
only — smallest safe diff that exposes the clean `SpatialTheoryBuilder` seam for the
Dyson / coupled‑propagator machinery.

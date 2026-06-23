# Prediagram visualization — `dd.plot_prediagrams` & `dd.prediagram_mappings`

Draw the **contributing prediagrams** for a theory at correlator order `k` and
loop order `max_ell`, and list how each specializes to the theory's field types.
Implemented in [`msrjd/diagrams/prediagram_plot.py`](../msrjd/diagrams/prediagram_plot.py);
surfaced on the `dd` API in [`notebooks/daedalus.py`](../notebooks/daedalus.py).

## Quick start

```python
import daedalus as dd
model, mod = dd.load_theory('ou_sextic')

dd.plot_prediagrams(model, k=2, max_ell=1)            # the grouped figure (returns a Figure)
dd.prediagram_mappings(model, k=2, max_ell=1)          # the per-diagram label maps (prints + returns)
```

`model` may be a model dict (from `load_theory`) or a theory-name string.
Practical range: **k + max_ell ≤ 4** (e.g. k=2,ℓ=2 = 71–78 diagrams; k=3,ℓ=1 = 20).

## What a prediagram is

A **prediagram** is a directed MSR-JD topology — vertices + directed propagators —
*before* the fields are assigned. `plot_prediagrams` shows exactly the prediagrams
that survive the theory's vertex/source filter (the topologies the theory can
actually realise), enumerated by `contributing_prediagrams(model, k, max_ell)` →
`{ell: [(D, G, leaves, internal), …]}` (each `D` a Sage `DiGraph`).

Roles are structural (`msrjd.diagrams.filter.classify_prediagram_vertices`):

| role               | rule                         | marker            | label        | position |
|--------------------|------------------------------|-------------------|--------------|----------|
| external leg       | leaf                         | ○ hollow circle   | `1, 2, …`    | left     |
| internal vertex    | non-leaf, in-degree > 0      | ● filled circle   | `a, b, c, …` | middle   |
| noise source       | non-leaf, in-degree 0        | ■ filled square   | `i, ii, …`   | right    |

Conventions:
- **Time flows right → left** (Buice/Ocker): sources on the right, external legs
  on the left.
- Each propagator carries a **mid-edge arrowhead** giving its direction (φ̃ → φ).
- **Propagators are named by their endpoints** — `a→b`, `i→1`, … — read straight
  off the labelled vertices; the two halves of a bubble are just `a→b` and `b→a`,
  and same-direction parallel edges get a subscript (`i→a₁`, `i→a₂`). There is no
  per-edge symbol on the figure, which keeps dense diagrams legible.
- Diagrams are **grouped by topology family** (a header band per family: number of
  internal vertices, number of sources, and the loop motif — bubble / sunset / …).

## The label-mapping tables

`prediagram_mappings(model, k, max_ell)` lists, for each prediagram (same numbering
as the figure), its typed realizations — how the generic labels specialize:

```
ℓ=1 · 1 internal, 2 sources · bubble · #1   (1 typing)
  ── typing 1 ──
     source i    → K^(2) ⟨xt xt⟩                       (noise cumulant on response legs)
     vertex a    → coeff 10*gamma*xstar1^2 + eps   …    (the interaction monomial)
     propagator i→a  → G[dx ← xt]                       (bare response propagator)
     external legs → 1=dx, 2=dx                          (the correlator's fields)
```

Returns `(result, text)`; `result` is `{ell: [entry, …]}` keeping every typing.
Options: `max_typings` (cap printed typings per prediagram), `external_fields`
(default = k copies of the first physical field), `use_propagator=True` (build the
propagator and prune identically-zero typings — the exact contributing set).

## Layout

Node positions come from **graphviz `dot`** (proper layered layout: rank
assignment, crossing-minimization, long-edge routing) via `pydot`, in the
right→left convention. If graphviz/pydot are unavailable the code falls back
automatically to a built-in SCC-aware layered layout (`_layout` tries
`_layout_graphviz`, excepts to `_layout_layered`).

Edge rendering on top of the node positions:
- An edge is drawn **straight** unless its segment would pass within ~0.32 of a
  node at a **strictly-intermediate rank**; only then does it arc, bowing *away*
  from the blocking node. Parallel edges (bubbles) always fan into a lens.
- The arc curvature sign follows matplotlib's `arc3` control point
  (`midpoint + rad·(dy, −dx)`); clearance ~constant (`rad ∝ 1/span`) so arcs
  don't balloon.

### Adaptive sizing

- **k ≤ 2**: 3 diagrams per row, compact cells.
- **k ≥ 3**: 2 per row with larger cells — k≥3 diagrams carry 3 external legs +
  3 sources (more fan-out), so they need ~2× the area to stay legible.

Override per call with `ncol=…` (e.g. `ncol=1` for one full-width diagram per
row). Node spacing is `nodesep=0.7, ranksep=1.05` in `dot`.

## Dependency (optional)

The clean layered layout needs the graphviz `dot` binary + `pydot`:

```
brew install graphviz                 # the dot binary (→ on PATH)
sage -python -m pip install pydot     # into Sage's python (the pipeline runs via sage -python)
```

Without them the figure still renders via the built-in fallback (uglier on dense
diagrams). See also the memory note `reference_graphviz_prediagram_viz`.

## Key functions (`msrjd/diagrams/prediagram_plot.py`)

| function | role |
|----------|------|
| `contributing_prediagrams(model, k, max_ell)` | enumerate + filter the topologies |
| `plot_prediagrams(model, k, max_ell, save=, ncol=)` | the grouped figure |
| `prediagram_mappings(model, k, max_ell, …)` | the typed label-map tables |
| `draw_prediagram(D, leaves, ax, title=)` | render one prediagram into an axis |
| `_generic_labels(D, leaves)` | layout + role labels (shared by figure & tables) |
| `topo_signature` / `sig_label` | topology grouping key + header text |

A reusable audit harness lives in `scratch/_audit_render.py` (paginated large
panels) and `scratch/_audit_geom.py` (programmatic node-overlap sweep); the
style-audit log is `scratch/_diagram_style_audit.md`.

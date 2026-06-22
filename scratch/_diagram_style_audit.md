# Prediagram figure — overnight clarity/style audit

Goal: go diagram by diagram across theories and (k, ℓ), find clarity/style
issues, fix them in `msrjd/diagrams/prediagram_plot.py`, commit incrementally.
User's standing complaint: edges still arc when a straight line would do.

## Method
- Render each (theory, k, ℓ) as large individual panels → `/tmp/audit_*.png`.
- Inspect, catalog issues below, fix, re-render, verify, commit.
- Keep tree/1-loop crisp; only arc when a node truly blocks the straight path.

## Cases to sweep
- ou_quartic   k=2 ℓ≤2
- ou_sextic    k=2 ℓ≤2 ; k=3 ℓ≤1
- ou_quartic   k=3 ℓ≤1 ; k=4 ℓ≤0
- (multifield) ou_quartic_two_dim_color_corr k=2 ℓ≤1  — variety / 2 fields

## Findings & fixes  (newest first)

### Round 1
- [FIX] Arc blocker filter used x-coordinate "between endpoints", which flagged
  nodes at the SAME rank as the target (a neighbour of v) as blockers → spurious
  arcs.  Changed to rank-strict: only nodes at a strictly-intermediate rank can
  block.  (commit pending)
- [CONFIRMED] surveyed ou_quartic k2ℓ2 (71), ou_sextic k3ℓ1 (20), ou_sextic
  k2ℓ2 (78) at 6/page large panels — only bubbles arc now, all other edges
  straight (incl. crossings drawn as straight X). Spurious-arc issue resolved.

### Round 2 (grouped-figure polish)
- [FIX] title "#n" overlapped the top vertex label (e.g. #2 on 'a').  Now the
  vertex labels are included in the extent (track) and the title pad lifted, so
  the title always clears.
- [FIX] excessive per-cell whitespace — the old draw margins were large
  (+0.95/+0.6).  Tightened to ~0.3 (labels are in the extent now), CELL 3.5→3.05,
  hspace 0.45→0.32, wspace 0.20→0.16.  Diagrams fill their cells.
- [FIX/STYLE] sources and internal vertices were both filled circles (only
  position/label distinguished them).  Sources are now filled SQUARES ■; legend
  updated (○ external · ● internal · ■ source).  3 roles read at a glance.
  NOTE: stylistic — easy to revert to a circle if the user prefers uniform dots.

### Round 3 (programmatic full sweep)
- [VERIFIED] `_audit_geom.py` scanned 191 diagrams (ou_quartic k2ℓ2/k3ℓ1/k4ℓ0 +
  ou_sextic k2ℓ2/k3ℓ1 — the whole topology space, field-agnostic): **0 node
  overlaps**; min node-node separation 1.17 (threshold 0.40).  Layout is
  geometrically sound everywhere.
- [VERIFIED] colored 2-field (ou_quartic_two_dim_color_corr) renders identically
  — topologies are field-agnostic; multi-field only affects the typings.

## Summary of overnight changes (all committed + pushed)
1. arcs only when a node BLOCKS the straight path (rank-strict blocker)  → b51bd95
2. title clearance + tighter cells + source marker ■ (3 distinct roles) → 375ca70
   (earlier this session: endpoint propagator naming, bigger diagrams,
    mid-edge arrowheads, edges touch vertices, arc sign fix)
Net: tree/1-loop are clean straights; only bubbles + truly-obstructed long edges
curve; 3 roles read at a glance; no overlaps; titles never collide.

## Decision log
- Did NOT cap marker size for sparse diagrams (would add whitespace the user
  disliked; trees-bigger is acceptable since grouped by complexity).
- Did NOT force consistent scale across groups (already consistent within a
  group; trees SHOULD render larger/clearer).

## Open ideas (style)
- consistent node size across diagrams (currently aspect-fit varies it)
- title (#n) can sit close to the top vertex label — pad/clearance
- bubble lens width vs node spacing
- arrowhead size relative to bigger cells

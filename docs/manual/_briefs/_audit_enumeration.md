# Audit: `sections/09-enumeration.tex` (Diagram Enumeration)

Adversarial accuracy + LaTeX-validity audit against:
- `msrjd/enumeration/loop_diagram_enumeration.py` (583 lines, read in full)
- `msrjd/enumeration/degree_scan.py` (135 lines, read in full)
- cross-checked: `msrjd/diagrams/type_assignment.py`, `docs/manual/daedalus_manual.tex` (preamble), `docs/manual/sections/02-quickstart.tex`, git log.

**Verdict: minor-issues.** `latex_ok = true` (chapter compiles; all environments balance, all macros/environments defined, no bare specials in prose). The vast majority of the chapter is accurate — every named identifier exists with the stated signature, and ~50 `:line` citations all land correctly. Two material issues and a few nits below.

---

## Findings

- **MAJOR — fabricated topology counts attributed to a code comment.**
  Manual claim: Gotcha (ll. 360–363) "The deduplicated *topology* counts — $(k,\ell)=(2,1)\to 9$, $(3,1)\to 67$, $(2,2)\to 289$ (`loop_diagram_enumeration.py:138` comment) — are the same before and after the fix." Repeated in the worked example (ll. 1126–1129): "this yields **9 topologies** — the anchor number recorded in the code comment at `loop_diagram_enumeration.py:138` … the comment also records $(3,1)\to 67$ and $(2,2)\to 289$."
  Code reality: the comment block at lines 129–138 contains **no** topology counts. Line 138 is `# (4,1),(2,3)}), but only the proven bound guarantees completeness.` `grep` for 9/67/289 in the file returns nothing. The comment records the *slack-verification orders* `{(2,1),(3,1),(2,2),(3,2),(4,1),(2,3)}`, not their topology counts. The numbers 9/67/289 may be correct combinatorially, but they are **not** "recorded in the code comment at :138" — that attribution is invented, and is stated twice.
  Location: `09-enumeration.tex` ll. 360–363 and 1126–1129; code `loop_diagram_enumeration.py:129–138`.

- **MAJOR (LaTeX quality, not a compile error) — `\ref{gotcha:...}` cross-references will not render the intended number.**
  Manual claim: the chapter labels gotchas (`\begin{gotcha}\label{gotcha:sentinel}` etc.) and refers to them with "Gotcha~\ref{gotcha:sentinel}", "Gotcha~\ref{gotcha:plus10}", "Gotcha~\ref{gotcha:double-build}", "Gotcha~\ref{gotcha:edgelabels}", "Gotcha~\ref{gotcha:leaffirst}", "Gotcha~\ref{gotcha:deadimport}", "Gotcha~\ref{gotcha:ell0-skip}" (7 distinct `\ref{gotcha:…}` sites). It assumes these resolve to a "Gotcha N" number.
  Code reality (preamble): `note`, `gotcha`, `defn` are plain `\newtcolorbox{...}[1][]` boxes (`daedalus_manual.tex:107–112`) with a *static* title and **no counter** — there is no `\tcbuselibrary{theorems}`, no `\newtcbtheorem`, no `use counter` anywhere in the preamble. A `\label` inside a counterless tcolorbox captures the *last stepped counter* (the enclosing section/equation), so `\ref{gotcha:sentinel}` prints e.g. a section number, not the gotcha's own index. Chapter 9 is the **only** manual file that uses `\ref{gotcha:…}` (no working precedent). The document still compiles (so `latex_ok` stays true), but every gotcha cross-reference shows a wrong/misleading number.
  Location: `09-enumeration.tex` (all `\ref{gotcha:*}` sites); preamble `daedalus_manual.tex:107–112`.

- **MINOR — dangling file reference: `scratch/bound_check.py` does not exist.**
  Manual claim: ll. 307 and 342 cite "`scratch/bound\_check.py`" as the location of the false-lemma counterexample script ("…recorded in the comment at `:133` (and in `scratch/bound\_check.py`)").
  Code reality: `scratch/bound_check.py` is not present on disk (`ls`/`wc`/`find` all confirm No such file). (It appears in the stale session-start git-status snapshot as untracked, but has since been removed, along with `theta_counterexamples.py` / `d_audit.py`.) Scratch files are inherently ephemeral; the load-bearing citation (the in-code comment at `:133`) is correct, so this is a low-severity stale pointer.
  Location: `09-enumeration.tex` ll. 307, 342.

- **MINOR — invented "$|V|=8$" detail on the counterexample, not in the source comment.**
  Manual claim: l. 342–344 "The counterexample … is *three doubled-edge bubbles on a hub*, $|V|=8$, where every decomposition has $j=3$ and $\theta = 3 > 2\ell-j-\lfloor j/3\rfloor$."
  Code reality: the comment (`:134`) says only "three doubled-edge bubbles on a hub — every decomposition has j = 3, theta = 3 > 2*ell - j - j//3". It does **not** state $|V|=8$. The $j=3$, $\theta=3$, and inequality all match the comment verbatim; the "$|V|=8$" is the chapter's own addition and is geometrically suspect (a hub + 3 satellite bubble-partners reads as $|V|=4$, not 8), so it is an unverified/likely-wrong embellishment of a claim the code does not make.
  Location: `09-enumeration.tex` l. 343; code `loop_diagram_enumeration.py:134`.

- **NIT — `\file{project\_enumeration\_bound\_fix.md}` is a memory note, not a repo file.**
  Manual claim: l. 307 cites "the project note `\file{project\_enumeration\_bound\_fix.md}`".
  Code reality: that file exists only under `~/.claude/.../memory/`, not in the repository tree. Wrapping it in `\file{}` (the inline-code/file macro) implies a repo path. It *does* exist as a referenced doc, so this is cosmetic.
  Location: `09-enumeration.tex` l. 307.

- **NIT — off-by-one inside the `ell=0` comment block.**
  Manual claim: ll. 528–529 attribute the phrase "the tree IS the topology — no contraction ambiguity" to `loop_diagram_enumeration.py:188`.
  Code reality: that phrase is on line **187**; line 188 is the continuation `# topology — no contraction ambiguity — so skip these checks.` Same 3-line comment block (186–188); trivially off.
  Location: `09-enumeration.tex` l. 529; code `:186–188`.

---

## Spot-checks that PASSED (so they are not re-litigated)

- Identifiers/signatures all exist exactly as described: `classify_vertices_sage`(:32), `has_adjacent_degree2_sage`(:41), `check_deg3_has_non_deg2_neighbor`(:50), `check_leaf_neighbors_not_all_deg2`(:58), `count_cycles_sage`(:68), `relabel_leaves_first`(:74), `graphs_isomorphic_with_labels`(:84), `directed_graphs_isomorphic_with_labels`(:99), `generate_trees_with_constraints`(:118), `generate_edge_multisets`(:166), `add_edges_to_tree`(:173), `check_topology_constraints`(:180), `process_tree_parallel`(:203), `_remove_isomorphic_undirected`(:219), `_enumerate_topologies_raw`(:228), `orient_edges`(:264), `check_orientation_constraints`(:275), `enumerate_orientations`(:295), `remove_isomorphic_directed`(:304), `_enumerate_prediagrams_raw`(:318), public API `enumerate_all`(:434)/`show_all`(:462)/`enumerate_trees`(:478)/`enumerate_topologies`(:498)/`enumerate_prediagrams`(:515)/`show_trees`(:535)/`show_topologies`(:552)/`show_prediagrams`(:569); `degree_scan` `max_vertex_degree`(:17)/`scan_source_vertices`(:45)/`check_taylor_order`(:70)/`ensure_taylor_order`(:91). The pasted code blocks match the source.
- Inline `:line` cites all correct: `j_max`(:124), `v3_max`(:128), `v2_max`(:139), `+10` cap(:143), `ell` gate(:189), relabel call(:213), edge-label add(:269–271), bit-extraction(:297), defaultdict(:349), `set_vertex`(:93), `is_isomorphic`(:96), `graphs.trees`(:149), `Graph(...multiedges...)`(:174), `DiGraph(...)`(:265), `ThreadPoolExecutor`(:248,:332), counts dict(:454), imports(:22–25), docstring "Requires a SageMath kernel"(:7); `degree_scan` imports(:13–14), stale-prop save block(:128–133).
- Behavioral claims correct: `enumerate_all` returns `(trees, topologies, prediagrams, counts)` with `counts={n_trees,n_topologies,n_prediagrams}`; `process_tree_parallel` arg tuple `(tree,j,num_leaves,k,ell)`; the five orientation constraints map 1:1 to `check_orientation_constraints`; `count_cycles_sage` returns `-1` sentinel for disconnected; `Counter` imported-but-unused (only `defaultdict` used); thread-only (no fork) parallelism with `n_threads=1` default; relabel uses `inplace=False`. Downstream contract verified against `type_assignment.py`: `enumerate_typed_diagrams(prediagram, …)` is the first-arg-prediagram consumer (:115), it unpacks `(D, G, leaves, internal)` (:36,:123,:591), and iterates `list(D.edges())  # 3-tuples (u, v, label)` (:142) — exactly as the chapter claims.
- Algebra correct: `v2_max = 2*v3_max+3ell-k-3j+3 = k+3ell-j-1` expansion; `j_max = ell+⌊ell/2⌋ = ⌊3ell/2⌋` and the tabulated values (ℓ=1→1, ℓ=2→3, ℓ=3→4); worked example (2,1): j=0 → v3_max=0,v2_max=4; j=1 → v3_max=1,v2_max=3.
- Commit `40454e7` ("enumeration: replace v2_max with the PROVEN tree-decomposition bound (drop −⌊j/3⌋)") is correct.
- Cross-refs resolve: `ch:type-assignment` (10-type-assignment.tex), `ch:quickstart` (02-quickstart.tex); the `[5/7] Enumerate prediagrams` phase label exists in 02-quickstart.tex. All intra-chapter `\label`/`\ref`/`\eqref` pairs are consistent (no dangling targets).
- LaTeX validity: environments balance (lstlisting 18/18, itemize 9/9, enumerate 3/3, description 3/3, align* 1/1, equation 6/6, defn 2/2, note 3/3, gotcha 11/11); `\code`,`\file`,`\term`,`\msrjd`,`\avg`,`\Sym` and the `defn`/`note`/`gotcha` boxes are all defined in `daedalus_manual.tex`; `hyperref` loaded (so `\texorpdfstring` is fine); no bare `_ # % & $` in prose (all such chars are inside math, listings, `\code{}`/`\file{}`, `align*` `&`, or header `%` comments).

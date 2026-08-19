"""
loop_diagram_enumeration.py
============================
Enumerate nonisomorphic trees, topologies, and prediagrams for
ell-loop corrections to k-point functions in MSRJD field theory.

Requires a SageMath kernel.

Public API
----------
enumerate_all(k, ell, ...)          -> (trees, topologies, prediagrams, counts)
enumerate_trees(k, ell, ...)        -> (trees, count)
enumerate_topologies(k, ell, ...)   -> (topologies, count)
enumerate_prediagrams(k, ell, ...)  -> (prediagrams, count)

show_all(k, ell, ...)               -> counts dict  (+ inline display)
show_trees(k, ell, ...)             -> count  (+ inline display)
show_topologies(k, ell, ...)        -> count
show_prediagrams(k, ell, ...)       -> count
"""

from sage.all import *
from itertools import combinations, combinations_with_replacement
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===========================================================================
# Core graph utilities
# ===========================================================================

def classify_vertices_sage(G):
    """Returns (leaves, internal, degree_2, degree_3plus)."""
    leaves      = [v for v in G.vertices() if G.degree(v) == 1]
    internal    = [v for v in G.vertices() if G.degree(v) > 1]
    degree_2    = [v for v in G.vertices() if G.degree(v) == 2]
    degree_3plus = [v for v in G.vertices() if G.degree(v) >= 3]
    return leaves, internal, degree_2, degree_3plus


def has_adjacent_degree2_sage(G, degree_2_vertices):
    degree_2_set = set(degree_2_vertices)
    for v in degree_2_vertices:
        for neighbor in G.neighbors(v):
            if neighbor in degree_2_set and neighbor != v:
                return True
    return False


def check_deg3_has_non_deg2_neighbor(G, deg3plus_vertices, deg2_vertices):
    deg2_set = set(deg2_vertices)
    for v in deg3plus_vertices:
        if all(n in deg2_set for n in G.neighbors(v)):
            return False
    return True


def check_leaf_neighbors_not_all_deg2(G, leaves, deg2_vertices):
    deg2_set = set(deg2_vertices)
    leaf_neighbors = set()
    for leaf in leaves:
        leaf_neighbors.update(G.neighbors(leaf))
    if leaf_neighbors and all(n in deg2_set for n in leaf_neighbors):
        return False
    return True


def count_cycles_sage(G):
    if not G.is_connected():
        return -1
    return G.size() - G.order() + 1


def relabel_leaves_first(G):
    leaves, internal, _, _ = classify_vertices_sage(G)
    relabel_dict = {}
    for idx, v in enumerate(sorted(leaves)):
        relabel_dict[v] = idx
    for idx, v in enumerate(sorted(internal)):
        relabel_dict[v] = len(leaves) + idx
    return G.relabel(relabel_dict, inplace=False), relabel_dict


def graphs_isomorphic_with_labels(G1, leaves1, G2, leaves2):
    if G1.order() != G2.order() or G1.size() != G2.size():
        return False
    if len(leaves1) != len(leaves2):
        return False
    leaves1_set = set(leaves1)
    leaves2_set = set(leaves2)
    G1c = G1.copy(); G2c = G2.copy()
    for v in G1.vertices():
        G1c.set_vertex(v, 0 if v in leaves1_set else 1)
    for v in G2.vertices():
        G2c.set_vertex(v, 0 if v in leaves2_set else 1)
    return G1c.is_isomorphic(G2c)


def directed_graphs_isomorphic_with_labels(D1, leaves1, D2, leaves2):
    if D1.order() != D2.order() or D1.size() != D2.size():
        return False
    if len(leaves1) != len(leaves2):
        return False
    leaves1_set = set(leaves1)
    leaves2_set = set(leaves2)
    D1c = D1.copy(); D2c = D2.copy()
    for v in D1.vertices():
        D1c.set_vertex(v, 0 if v in leaves1_set else 1)
    for v in D2.vertices():
        D2c.set_vertex(v, 0 if v in leaves2_set else 1)
    return D1c.is_isomorphic(D2c)


# ===========================================================================
# Tree generation
# ===========================================================================

def v_max_orientable_bound(k, ell):
    """Orientability cap on |V|:  |V| <= 3k + 3*ell - 3.

    Derivation (see the block comment in ``generate_trees_with_constraints``):
    sum(deg) = 2|E| = 2(|V|-1+ell) with sum(deg) >= k + 2|V2^G| + 3|V3^G| gives
    |V3^G| <= k + 2*ell - 2, which with |V2^G| <= k + ell - 1 (Lemma v2g)
    yields |V| <= 3k + 3*ell - 3.

    Factored out of the tree loop so ``tests/test_orientability_cap.py`` can
    monkeypatch it to +infinity and assert that lifting the cap adds no
    prediagram -- i.e. that every topology above it really is non-orientable.
    """
    return 3 * k + 3 * ell - 3


def generate_trees_with_constraints(k, ell, max_vertices_search=50):
    """
    Generate all trees T with k+j leaves satisfying degree constraints.
    Returns list of (tree, j, num_leaves).
    """
    valid_trees = []
    j_max = ell + (ell // 2)

    for j in range(0, j_max + 1):
        num_leaves = k + j
        v3_max = k + j - 2
        # |V2^T| <= k + 3*ell - j - 1 (= 2*v3_max + 3*ell - k - 3*j + 3): the
        # PROVEN tree-decomposition bound (paper appendix, corrected theorem:
        # degree-partition identity + |V2^G| <= k + ell - 1 from orientability
        # + theta <= 2*ell - j by incidence counting).  The earlier extra
        # "- (j // 3)" tightening came from a theta-lemma that is FALSE at
        # ell >= 3 (counterexample: three doubled-edge bubbles on a hub —
        # every decomposition has j = 3, theta = 3 > 2*ell - j - j//3); it
        # never happened to bind at small orders (slack >= 2 verified by
        # exhaustive enumeration at (k,ell) in {(2,1),(3,1),(2,2),(3,2),
        # (4,1),(2,3)}), but only the proven bound guarantees completeness.
        v2_max = 2 * v3_max + 3 * ell - k - 3 * j + 3

        min_n = num_leaves if num_leaves == 1 else num_leaves + 1
        # Only the PROVEN vertex-count bound (num_leaves + v2_max + v3_max) and
        # the global search cap (max_vertices_search=50) apply here; the per-tree
        # degree constraints below (v3_max / v2_max) do the rest.  An earlier
        # extra clamp ``min(max_n, num_leaves + min(10, v2_max + v3_max))`` was
        # un-proven (it silently truncated the search at high vertex counts and
        # could drop valid trees at loop order >= 3); removed to restore
        # completeness.  Inert at ell <= 2 (the proven bound already binds).
        # Orientability caps |V| independently of the tree decomposition.
        # Partitioning G by degree, sum(deg) = 2|E| = 2(|V|-1+ell) and
        # sum(deg) >= k + 2|V2^G| + 3|V3^G| give
        #     |V3^G| <= k + 2*ell - 2 ,
        # which with |V2^G| <= k + ell - 1 (Lemma v2g, from orientability)
        # yields
        #     |V| = k + |V2^G| + |V3^G| <= 3k + 3*ell - 3 .
        # T and G share a vertex set, so a larger tree can only ever produce
        # NON-ORIENTABLE topologies -- verified exhaustively at (k,ell) in
        # {(2,1),(3,1),(4,1),(2,2)}: every topology above the cap admits zero
        # valid orientations, so no prediagram is lost.  Both bounds are tight
        # (attained at each of those points).  This is what makes ell=3
        # tractable: it removes 33x of the candidate multisets there.
        v_max_orientable = v_max_orientable_bound(k, ell)
        max_n = min(num_leaves + v2_max + v3_max, v_max_orientable,
                    max_vertices_search)

        if max_n < min_n:
            continue

        for n in range(min_n, max_n + 1):
            for tree in graphs.trees(n):
                leaves, internal, degree_2, degree_3plus = classify_vertices_sage(tree)
                if len(leaves) != num_leaves:
                    continue
                if len(degree_3plus) > v3_max:
                    continue
                if len(degree_2) > v2_max:
                    continue
                valid_trees.append((tree, j, num_leaves))

    return valid_trees


# ===========================================================================
# Edge addition
# ===========================================================================

def generate_edge_multisets(all_vertices, ell):
    if len(all_vertices) < 2:
        return []
    possible_edges = list(combinations(all_vertices, 2))
    return list(combinations_with_replacement(possible_edges, ell))


def add_edges_to_tree(tree, edge_multiset):
    G = Graph(tree, multiedges=True, loops=False)
    for edge in edge_multiset:
        G.add_edge(edge[0], edge[1])
    return G


def check_topology_constraints(G, k, ell=None):
    leaves, internal, degree_2, degree_3plus = classify_vertices_sage(G)
    if len(leaves) != k:
        return False
    if not G.is_connected():
        return False
    # Degree-2 pruning constraints avoid redundant topologies when edges
    # will be added (ell >= 1).  At tree level (ell=0) the tree IS the
    # topology — no contraction ambiguity — so skip these checks.
    if ell is None or ell > 0:
        if has_adjacent_degree2_sage(G, degree_2):
            return False
        if not check_deg3_has_non_deg2_neighbor(G, degree_3plus, degree_2):
            return False
        if not check_leaf_neighbors_not_all_deg2(G, leaves, degree_2):
            return False
    return True


# ===========================================================================
# Topology enumeration
# ===========================================================================

def process_tree_parallel(args):
    tree, j, num_leaves, k, ell = args
    local_candidates = []
    all_verts = list(tree.vertices())

    # Base-tree invariants, computed once, for the pre-build screens.
    base_deg = {v: tree.degree(v) for v in all_verts}
    tree_leaves = frozenset(v for v, d in base_deg.items() if d == 1)
    base_d2 = frozenset(v for v, d in base_deg.items() if d == 2)
    tree_edges = list(tree.edges(labels=False))

    # Restrict the pair pool before enumerating multisets.  Exactly j = the
    # tree's surplus leaves must be killed, and touching a leaf always kills
    # it, so a pair carrying more than j leaves can never appear in a
    # qualifying multiset.  At j = 0 this drops every leaf-incident pair and
    # leaves only internal-internal edges.
    n_surplus = num_leaves - k
    pairs = [(u, v) for u, v in combinations(all_verts, 2)
             if (u in tree_leaves) + (v in tree_leaves) <= n_surplus]

    for edge_multiset in combinations_with_replacement(pairs, ell):
        # ── Screen 1: leaf count, in O(ell) ──────────────────────────
        # Degrees only ever increase, so no new leaf can appear and a tree
        # leaf survives iff the multiset does not touch it:
        #     leaves(G) = leaves(T) - |touched cap leaves(T)| .
        # Touching only the ell added edges avoids copying the whole degree
        # dict per multiset -- this screen runs on every multiset and
        # rejects ~80% of them, so its constant factor dominates.
        delta = {}
        for u, v in edge_multiset:
            delta[u] = delta.get(u, 0) + 1
            delta[v] = delta.get(v, 0) + 1
        if num_leaves - sum(1 for v in delta if v in tree_leaves) != k:
            continue

        # ── Screen 2: adjacent degree-2 vertices (Thm. no-adj-2) ─────
        # Rejects ~93% of what survives screen 1.  Only touched vertices can
        # change degree, so the degree-2 set updates from ``base_d2`` in
        # O(ell); adjacency is the tree's edges together with the multiset.
        # NOTE: the degree-2 constraints apply only for ell > 0.  At tree
        # level the tree IS the topology -- there is no contraction ambiguity
        # to break -- and ``check_topology_constraints`` skips them, so
        # screening on them here would wrongly discard every tree-level
        # record.
        d2 = set(base_d2)
        for v, dv in delta.items():
            if base_deg[v] + dv == 2:
                d2.add(v)
            else:
                d2.discard(v)
        if ell > 0 and len(d2) >= 2:
            if any(u in d2 and v in d2 for u, v in tree_edges) or \
               any(u in d2 and v in d2 for u, v in edge_multiset):
                continue

        G = add_edges_to_tree(tree, edge_multiset)
        # NOTE: the former ``count_cycles_sage(G) != ell`` guard here was
        # dead.  ``generate_edge_multisets`` draws from ``combinations(V, 2)``
        # so no edge is a self-loop, and a connected tree with V-1 edges plus
        # ell of them has b1 = (V-1+ell) - V + 1 = ell identically.  The guard
        # never fired but paid a full ``is_connected()`` per candidate, which
        # ``check_topology_constraints`` then recomputed.

        # The screens above already determined every degree, so the vertex
        # classification is in hand: re-deriving it via
        # ``classify_vertices_sage`` (which ``check_topology_constraints`` and
        # ``relabel_leaves_first`` each did independently) is redundant.  The
        # leaf-count and connectivity tests inside
        # ``check_topology_constraints`` are likewise already settled -- the
        # first by screen 1, the second because tree-plus-edges is always
        # connected -- as is the adjacent-degree-2 test by screen 2.  Only the
        # last two structural checks remain.
        leaves = [v for v in all_verts if base_deg[v] + delta.get(v, 0) == 1]
        if ell > 0:
            d3 = [v for v in all_verts if base_deg[v] + delta.get(v, 0) >= 3]
            if not check_deg3_has_non_deg2_neighbor(G, d3, d2):
                continue
            if not check_leaf_neighbors_not_all_deg2(G, leaves, d2):
                continue

        G_relabeled, _ = relabel_leaves_first(G)
        # ``relabel_leaves_first`` maps the sorted leaves onto 0..|L|-1 and the
        # sorted internal vertices onto |L|..|V|-1, so the post-relabel
        # classification is these ranges by construction.
        n_leaf = len(leaves)
        leaves_final = list(range(n_leaf))
        internal_final = list(range(n_leaf, len(all_verts)))
        local_candidates.append((G_relabeled, leaves_final, internal_final))
    return local_candidates


def _iso_cert(G):
    """Hashable isomorphism certificate for a (possibly multi-) graph.

    One ``canonical_label()`` call replaces the O(N*U) pairwise
    ``is_isomorphic`` scan the dedup loops used to run: candidates whose
    certificates collide are isomorphic, so a dict keyed by the certificate
    deduplicates in O(N).

    Edge labels are stripped first.  ``orient_edges`` attaches a DISTINCT
    integer label to every edge copy, and ``canonical_label`` honours edge
    labels -- so canonicalising the labelled digraph makes every orientation
    look unique and defeats the dedup entirely.  ``is_isomorphic`` (used by
    the predicates this replaces) ignores edge labels by default, so
    stripping restores exactly the old equivalence.

    Vertex colouring by leaf/internal is deliberately NOT part of the
    certificate: leaves are exactly the degree-1 vertices and degree is an
    isomorphism invariant, so the ``set_vertex`` colouring the pairwise
    predicates applied was redundant (and ignored by ``is_isomorphic``).
    """
    cls = DiGraph if G.is_directed() else Graph
    stripped = cls(multiedges=True, loops=False)
    stripped.add_vertices(G.vertices())
    for u, v in G.edges(labels=False):
        stripped.add_edge(u, v)
    C = stripped.canonical_label()
    return (C.order(), tuple(sorted(C.edges(labels=False))))


def cert_to_graph(cert, directed):
    """Inverse of :func:`_iso_cert`: rebuild the canonical representative.

    ``_iso_cert`` returns ``(order, sorted_edges)`` of ``canonical_label()``,
    which is a LOSSLESS encoding of the graph up to isomorphism -- the pair is
    enough to reconstruct the canonical representative outright.  Repeated
    pairs in ``sorted_edges`` carry edge multiplicity, since
    ``edges(labels=False)`` lists every copy.

    This is what lets the enumeration pass ~100-byte tuples between stages and
    between processes instead of Sage graph objects, which cost ~13.8 kB of
    live memory apiece (measured on the (4,2) cache: 6.7 MB on disk, 0.91 GB
    resident).  Round-trip identity ``_iso_cert(cert_to_graph(c)) == c`` is
    asserted over every cached cell by ``tests/test_cert_roundtrip.py``.

    The leaf set is NOT stored: leaves are exactly the degree-1 vertices and
    degree is an isomorphism invariant, so ``leaves_of`` recovers it.
    """
    order, edges = cert
    cls = DiGraph if directed else Graph
    G = cls(multiedges=True, loops=False)
    G.add_vertices(range(order))
    for u, v in edges:
        G.add_edge(u, v)
    return G


def leaves_of(G):
    """External vertices of a (pre)diagram: exactly its degree-1 vertices."""
    return sorted(v for v in G.vertices() if G.degree(v) == 1)


def _remove_isomorphic_undirected(candidates):
    seen = {}
    for G, leaves, internal in candidates:
        seen.setdefault(_iso_cert(G), (G, leaves, internal))
    return list(seen.values())


def _enumerate_topologies_raw(k, ell, n_threads=1, max_vertices_search=50, verbose=True,
                              trees_with_j=None):
    """Internal: returns unique_multigraphs list."""
    if trees_with_j is None:
        trees_with_j = generate_trees_with_constraints(k, ell, max_vertices_search)

    if verbose:
        print(f"  {len(trees_with_j)} valid trees found")

    candidates = []

    if n_threads == 1:
        for i, (tree, j, num_leaves) in enumerate(trees_with_j):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing tree {i+1}/{len(trees_with_j)}...", end='\r')
            candidates.extend(process_tree_parallel((tree, j, num_leaves, k, ell)))
        if verbose:
            print(f"  Processed {len(trees_with_j)}/{len(trees_with_j)} trees. Done!")
    else:
        args_list = [(t, j, nl, k, ell) for t, j, nl in trees_with_j]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for future in as_completed(
                    [ex.submit(process_tree_parallel, a) for a in args_list]):
                candidates.extend(future.result())

    if verbose:
        print(f"  {len(candidates)} candidate topologies before isomorphism removal")

    unique = _remove_isomorphic_undirected(candidates)
    return unique


# ===========================================================================
# Orientation enumeration
# ===========================================================================

def orient_edges(G, orientation_bits):
    D = DiGraph(multiedges=True, loops=False)
    D.add_vertices(G.vertices())
    for i, (u, v) in enumerate(G.edges(labels=False)):
        if orientation_bits[i] == 0:
            D.add_edge(u, v, i)
        else:
            D.add_edge(v, u, i)
    return D


def check_orientation_constraints(D, leaves):
    leaves_set = set(leaves)
    if not D.is_directed_acyclic():
        return False
    for v in D.vertices():
        in_deg  = D.in_degree(v)
        out_deg = D.out_degree(v)
        if (in_deg + out_deg) == 1 and in_deg != 1:
            return False
        if out_deg == 0 and in_deg >= 2:
            return False
        if in_deg == 1 and out_deg == 1:
            return False
    sources = [v for v in D.vertices() if D.in_degree(v) == 0]
    for v in sources:
        if any(u in sources for u in D.neighbors_out(v)):
            return False
    return True


def enumerate_orientations(G, leaves):
    """Valid orientations of ``G``, branching only on the UNDETERMINED edges.

    Most edge directions are fixed before any search, by two of the causal
    Feynman rules:

      * (R2) every degree-1 vertex has in-degree 1, so an edge incident to a
        leaf is directed INTO the leaf;
      * (Corollary, paper appendix) every degree-2 vertex is a source with
        (in, out) = (0, 2), so both of its edge copies are directed OUT of it.

    These never conflict on a valid topology: a degree-2 vertex adjacent to a
    leaf forces the same direction from both ends, while adjacent leaves (a
    leaf cannot be a source) and adjacent degree-2 vertices (Theorem
    ``no adjacent degree-2``) are excluded at the topology level.  The only
    free edges are therefore those with BOTH endpoints internal of degree >= 3.

    This is a pure search-space restriction, not a semantic change:
    ``check_orientation_constraints`` already rejects every orientation that
    violates the forcings above (a degree-2 vertex oriented (2,0) trips the
    no-internal-sink test, (1,1) trips the bilinear test; a leaf oriented
    (0,1) trips the external-leg test), so the skipped candidates would have
    been discarded anyway and the returned set is unchanged.  Measured
    71-100% of edges forced, i.e. a 4x-550x smaller branch.
    """
    edges = list(G.edges(labels=False))
    leafset = set(leaves)
    deg = {v: G.degree(v) for v in G.vertices()}

    # ``orient_edges`` reads bit 0 as u -> v and bit 1 as v -> u.
    forced = {}
    for i, (u, v) in enumerate(edges):
        head_v = (v in leafset) or (deg[u] == 2)      # u -> v
        head_u = (u in leafset) or (deg[v] == 2)      # v -> u
        if head_v and head_u:
            return []          # contradictory forcing: no legal orientation
        if head_v:
            forced[i] = 0
        elif head_u:
            forced[i] = 1

    free = [i for i in range(len(edges)) if i not in forced]

    base = [0] * len(edges)
    for i, b in forced.items():
        base[i] = b

    out = []
    for bits in range(2 ** len(free)):
        pattern = list(base)
        for slot, i in enumerate(free):
            pattern[i] = (bits >> slot) & 1
        D = orient_edges(G, pattern)       # built once, not twice
        if check_orientation_constraints(D, leaves):
            out.append(D)
    return out


def remove_isomorphic_directed(directed_diagrams):
    seen = {}
    for D, G, leaves, internal in directed_diagrams:
        seen.setdefault(_iso_cert(D), (D, G, leaves, internal))
    return list(seen.values())


def process_orientation_parallel(args):
    G, leaves, internal = args
    return [(D, G, leaves, internal) for D in enumerate_orientations(G, leaves)]


def _enumerate_prediagrams_raw(topologies, n_threads=1, verbose=True):
    """Internal: returns unique_directed_diagrams list."""
    all_directed = []

    if n_threads == 1:
        for i, (G, leaves, internal) in enumerate(topologies):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing topology {i+1}/{len(topologies)}...", end='\r')
            for D in enumerate_orientations(G, leaves):
                all_directed.append((D, G, leaves, internal))
        if verbose:
            print(f"  Processed {len(topologies)}/{len(topologies)} topologies. Done!")
    else:
        args_list = [(G, leaves, internal) for G, leaves, internal in topologies]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for future in as_completed(
                    [ex.submit(process_orientation_parallel, a) for a in args_list]):
                all_directed.extend(future.result())

    if verbose:
        print(f"  {len(all_directed)} oriented diagrams before isomorphism removal")

    return remove_isomorphic_directed(all_directed)


# ===========================================================================
# Visualization helpers
# ===========================================================================

def _plot_trees(trees_with_j, k, ell, save=False):
    from sage.plot.plot import graphics_array as ga_func
    trees_by_j = defaultdict(list)
    for tree, j, num_leaves in trees_with_j:
        trees_by_j[j].append((tree, num_leaves))

    for j in sorted(trees_by_j.keys()):
        group = trees_by_j[j]
        print(f"\nTrees with j={j}  (k+j={k+j} leaves): {len(group)}")
        n_cols = min(4, len(group))
        n_rows = (len(group) + n_cols - 1) // n_cols
        plots = []
        for idx, (tree, num_leaves) in enumerate(group):
            leaves, internal, deg2, deg3 = classify_vertices_sage(tree)
            color_map = {}
            for v in tree.vertices():
                if v in set(leaves):
                    color_map.setdefault('black', []).append(v)
                elif v in set(deg2):
                    color_map.setdefault('lightblue', []).append(v)
                else:
                    color_map.setdefault('red', []).append(v)
            title = (f"Tree {idx+1}\n"
                     f"|V|={tree.order()}, |L|={len(leaves)}\n"
                     f"|V₂|={len(deg2)}, |V₃|={len(deg3)}")
            plots.append(tree.plot(vertex_colors=color_map, vertex_size=300,
                                   edge_thickness=2, title=title,
                                   title_pos=(0.5, -0.1)))
        ga = ga_func(plots, n_rows, n_cols)
        if save:
            ga.save(f'trees_j{j}_{ell}loop_{k}point.png',
                    figsize=[5 * n_cols, 4 * n_rows], dpi=150)
        ga.show(figsize=[5 * n_cols, 4 * n_rows])


def _plot_topologies(topologies, k, ell, save=False):
    from sage.plot.plot import graphics_array as ga_func
    n = len(topologies)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    plots = []
    for i, (G, leaves, internal) in enumerate(topologies):
        leaves_set = set(leaves)
        color_map = {}
        for v in G.vertices():
            key = 'black' if v in leaves_set else 'lightgray'
            color_map.setdefault(key, []).append(v)
        plots.append(G.plot(vertex_colors=color_map, vertex_size=400,
                            edge_thickness=2, title=f"Topology {i+1}"))
    ga = ga_func(plots, n_rows, n_cols)
    if save:
        ga.save(f'topologies_{ell}_loop_{k}_point.png',
                figsize=[6 * n_cols, 5 * n_rows], dpi=150)
    ga.show(figsize=[6 * n_cols, 5 * n_rows])


def _plot_prediagrams(prediagrams, k, ell, save=False):
    from sage.plot.plot import graphics_array as ga_func
    n = len(prediagrams)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    plots = []
    for i, (D, G, leaves, internal) in enumerate(prediagrams):
        leaves_set = set(leaves)
        sources = [v for v in D.vertices() if D.in_degree(v) == 0]
        color_map = {}
        for v in D.vertices():
            if v in leaves_set:
                color_map.setdefault('black', []).append(v)
            elif v in sources:
                color_map.setdefault('red', []).append(v)
            else:
                color_map.setdefault('lightblue', []).append(v)
        plots.append(D.plot(vertex_colors=color_map, vertex_size=400,
                            edge_thickness=2, edge_labels=False,
                            title=f"Prediagram {i+1}"))
    ga = ga_func(plots, n_rows, n_cols)
    if save:
        ga.save(f'prediagrams_{ell}_loop_{k}_point.png',
                figsize=[6 * n_cols, 5 * n_rows], dpi=150)
    ga.show(figsize=[6 * n_cols, 5 * n_rows])


# ===========================================================================
# Public API
# ===========================================================================

def enumerate_all(k, ell, n_threads=1, max_vertices_search=50, verbose=True):
    """
    Run the full enumeration pipeline once: trees → topologies → prediagrams.
    Tree generation is shared across all three stages — no repeated work.

    Returns
    -------
    trees       : list of (tree, j, num_leaves)
    topologies  : list of (G, leaves, internal)
    prediagrams : list of (D, G, leaves, internal)
    counts      : dict with keys n_trees, n_topologies, n_prediagrams
    """
    if verbose:
        print(f"Enumerating all for k={k}, ell={ell}...")

    trees = generate_trees_with_constraints(k, ell, max_vertices_search)
    topos = _enumerate_topologies_raw(k, ell, n_threads, max_vertices_search, verbose,
                                      trees_with_j=trees)
    pds   = _enumerate_prediagrams_raw(topos, n_threads, verbose)

    counts = dict(n_trees=len(trees), n_topologies=len(topos), n_prediagrams=len(pds))
    if verbose:
        print(f"\n  Trees: {counts['n_trees']}")
        print(f"  Topologies: {counts['n_topologies']}")
        print(f"  Prediagrams: {counts['n_prediagrams']}")
    return trees, topos, pds, counts


def show_all(k, ell, n_threads=1, max_vertices_search=50, save=False):
    """
    Run the full pipeline once, then display trees, topologies, and prediagrams.

    Returns
    -------
    counts : dict with keys n_trees, n_topologies, n_prediagrams
    """
    trees, topos, pds, counts = enumerate_all(k, ell, n_threads, max_vertices_search,
                                              verbose=True)
    if counts['n_trees']      > 0: _plot_trees(trees, k, ell, save=save)
    if counts['n_topologies'] > 0: _plot_topologies(topos, k, ell, save=save)
    if counts['n_prediagrams'] > 0: _plot_prediagrams(pds, k, ell, save=save)
    return counts


def enumerate_trees(k, ell, max_vertices_search=50, verbose=False):
    """
    Return all nonisomorphic trees for the (k, ell) problem.

    Parameters
    ----------
    k : int   k-point function
    ell : int   ell-loop correction

    Returns
    -------
    trees : list of (tree, j, num_leaves)
    count : int
    """
    trees = generate_trees_with_constraints(k, ell, max_vertices_search)
    if verbose:
        print(f"Trees for k={k}, ell={ell}: {len(trees)}")
    return trees, len(trees)


def enumerate_topologies(k, ell, n_threads=1, max_vertices_search=50, verbose=True):
    """
    Return all nonisomorphic undirected topologies for the (k, ell) problem.

    Returns
    -------
    topologies : list of (G, leaves, internal)
    count : int
    """
    if verbose:
        print(f"Enumerating topologies for k={k}, ell={ell}...")
    topo = _enumerate_topologies_raw(k, ell, n_threads, max_vertices_search, verbose)
    if verbose:
        print(f"\nNonisomorphic topologies: {len(topo)}")
    return topo, len(topo)


def enumerate_prediagrams(k, ell, n_threads=1, max_vertices_search=50, verbose=True):
    """
    Return all nonisomorphic prediagrams (oriented topologies) for (k, ell).

    Returns
    -------
    prediagrams : list of (D, G, leaves, internal)
    count : int
    """
    if verbose:
        print(f"Enumerating topologies for k={k}, ell={ell}...")
    topo = _enumerate_topologies_raw(k, ell, n_threads, max_vertices_search, verbose)
    if verbose:
        print(f"\n  {len(topo)} topologies. Now enumerating orientations...")
    pds = _enumerate_prediagrams_raw(topo, n_threads, verbose)
    if verbose:
        print(f"\nNonisomorphic prediagrams: {len(pds)}")
    return pds, len(pds)


def show_trees(k, ell, max_vertices_search=50, save=False):
    """
    Display all nonisomorphic trees for (k, ell) and print the count.

    Color code: black = leaves, lightblue = degree-2, red = degree-3+

    Returns
    -------
    count : int
    """
    trees, count = enumerate_trees(k, ell, max_vertices_search, verbose=False)
    print(f"Nonisomorphic trees for k={k}, ell={ell}: {count}")
    if count > 0:
        _plot_trees(trees, k, ell, save=save)
    return count


def show_topologies(k, ell, n_threads=1, max_vertices_search=50, save=False):
    """
    Display all nonisomorphic undirected topologies for (k, ell) and print the count.

    Color code: black = leaves, lightgray = internal vertices

    Returns
    -------
    count : int
    """
    topo, count = enumerate_topologies(k, ell, n_threads, max_vertices_search, verbose=True)
    print(f"\nNonisomorphic topologies for k={k}, ell={ell}: {count}")
    if count > 0:
        _plot_topologies(topo, k, ell, save=save)
    return count


def show_prediagrams(k, ell, n_threads=1, max_vertices_search=50, save=False):
    """
    Display all nonisomorphic prediagrams (oriented) for (k, ell) and print the count.

    Color code: black = external legs, red = sources, lightblue = internal vertices

    Returns
    -------
    count : int
    """
    pds, count = enumerate_prediagrams(k, ell, n_threads, max_vertices_search, verbose=True)
    print(f"\nNonisomorphic prediagrams for k={k}, ell={ell}: {count}")
    if count > 0:
        _plot_prediagrams(pds, k, ell, save=save)
    return count

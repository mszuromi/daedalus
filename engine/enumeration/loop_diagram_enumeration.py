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
        max_n = min(num_leaves + v2_max + v3_max, max_vertices_search)
        max_n = min(max_n, num_leaves + min(10, v2_max + v3_max))

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
    for edge_multiset in generate_edge_multisets(all_verts, ell):
        G = add_edges_to_tree(tree, edge_multiset)
        if count_cycles_sage(G) != ell:
            continue
        if not check_topology_constraints(G, k, ell=ell):
            continue
        G_relabeled, _ = relabel_leaves_first(G)
        leaves_final, internal_final, _, _ = classify_vertices_sage(G_relabeled)
        local_candidates.append((G_relabeled, leaves_final, internal_final))
    return local_candidates


def _remove_isomorphic_undirected(candidates):
    unique = []
    for G, leaves, internal in candidates:
        if not any(graphs_isomorphic_with_labels(G, leaves, G2, l2)
                   for G2, l2, _ in unique):
            unique.append((G, leaves, internal))
    return unique


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
    edges = list(G.edges(labels=False))
    return [orient_edges(G, [(bits >> i) & 1 for i in range(len(edges))])
            for bits in range(2 ** len(edges))
            if check_orientation_constraints(
                orient_edges(G, [(bits >> i) & 1 for i in range(len(edges))]),
                leaves)]


def remove_isomorphic_directed(directed_diagrams):
    unique = []
    for D, G, leaves, internal in directed_diagrams:
        if not any(directed_graphs_isomorphic_with_labels(D, leaves, D2, l2)
                   for D2, G2, l2, _ in unique):
            unique.append((D, G, leaves, internal))
    return unique


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

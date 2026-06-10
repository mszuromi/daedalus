"""Pure-combinatorics Wick-count oracle for the k=3, 1-loop, a^3 sector
of OU + a*x^2 (+ b*x^3) — no integrals.

Sector content: 3 externals (x at equal times), 3 interaction vertices
A_i = a*xt*x^2 (legs: 1 xtilde + 2 x), 3 noise vertices N_i = T*xt^2
(legs: 2 xtilde).  Every Wick pairing matches the 9 xtilde slots to the
9 x slots; each pairing's integrand is a product of retarded
propagators, so at EQUAL external times two pairings with isomorphic
(unpinned) digraphs integrate to the same value I0(class).

Therefore the per-class weight in I0 units is

    wick_w(class) = N_pairings(class) / (3! * 3!)

(the 1/n! from expanding exp(-S) per repeated vertex type), to be
compared against the pipeline's

    pipe_w(class) = sum over unique typed diagrams in the class of
                    S(Gamma) * n_mappings / comp,   n_mappings = 3!

If every class matches, the symmetry/compensation layer is exact for
this sector and the residual must come from the integrals; a mismatch
pinpoints the defective class and the factor.
"""
import sys
sys.path.insert(0, '.')
import itertools
from collections import Counter, defaultdict
from math import factorial

from sage.all import DiGraph, SR

# ── 1. Wick enumeration over collapsed digraphs ─────────────────────
# Nodes: A0,A1,A2 (color A), N0,N1,N2 (color N), E0,E1,E2 (color E).
# xtilde slots (tail of retarded edge: from the slot's vertex):
#   A_i contributes 1, N_i contributes 2.
# x slots (head): A_i contributes 2, E_j contributes 1.
TILDE = [('A', 0), ('A', 1), ('A', 2),
         ('N', 0), ('N', 0), ('N', 1), ('N', 1), ('N', 2), ('N', 2)]
XSLOT = [('A', 0), ('A', 0), ('A', 1), ('A', 1), ('A', 2), ('A', 2),
         ('E', 0), ('E', 1), ('E', 2)]

NODES = [('A', i) for i in range(3)] + [('N', i) for i in range(3)] \
      + [('E', i) for i in range(3)]


def classify_edges(edges):
    """edges: tuple of (tail_node, head_node).  Return None if the
    pairing is zero/excluded, else a canonical class key."""
    # G_R(0) = 0: xtilde and x on the SAME interaction vertex
    for u, v in edges:
        if u == v:
            return None
    # connectivity (undirected) over the 9 nodes
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen = set()
    stack = [NODES[0]]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj[n] - seen)
    if len(seen) != len(NODES):
        return None
    # directed cycle => product of Heavisides vanishes
    indeg = Counter()
    out = defaultdict(list)
    for u, v in edges:
        out[u].append(v)
        indeg[v] += 0  # touch
        indeg[v] += 1
    # Kahn
    indeg2 = Counter(indeg)
    q = [n for n in NODES if indeg2[n] == 0]
    npop = 0
    while q:
        n = q.pop()
        npop += 1
        for w in out[n]:
            indeg2[w] -= 1
            if indeg2[w] == 0:
                q.append(w)
    if npop != len(NODES):
        return None
    return canonical_key(edges)


_canon_cache = {}


def canonical_key(edges):
    ek = tuple(sorted(edges))
    hit = _canon_cache.get(ek)
    if hit is not None:
        return hit
    # Sage digraph with multiedges; color classes A / N / E.
    D = DiGraph(multiedges=True, loops=False)
    D.add_vertices(NODES)
    D.add_edges([(u, v) for u, v in edges])
    partition = [[n for n in NODES if n[0] == 'A'],
                 [n for n in NODES if n[0] == 'N'],
                 [n for n in NODES if n[0] == 'E']]
    C = D.canonical_label(partition=partition)
    key = str(sorted(C.edges(labels=False)))
    _canon_cache[ek] = key
    return key


REPS = {}


def enumerate_wick():
    counts = Counter()
    n_slots = len(XSLOT)
    for perm in itertools.permutations(range(n_slots)):
        edges = tuple((TILDE[i], XSLOT[perm[i]]) for i in range(n_slots))
        key = classify_edges(edges)
        if key is not None:
            counts[key] += 1
            REPS.setdefault(key, edges)
    return counts


print('enumerating 9! = 362880 Wick pairings ...')
wick_counts = enumerate_wick()
print('connected acyclic classes: %d' % len(wick_counts))
norm = factorial(3) * factorial(3)   # 1/3! per repeated vertex type
wick_w = {k: c / norm for k, c in wick_counts.items()}

# ── 2. Pipeline side ────────────────────────────────────────────────
from msrjd.core.field_theory import FieldTheory
from pipeline._propagator import build_propagator, compute_poles_and_residues
from pipeline._diagrams import enumerate_unique_diagrams
from msrjd.diagrams.symmetry import (
    classify_coefficient_factors, combinatorial_factor,
    external_wick_compensation,
)
from msrjd.diagrams.type_assignment import build_field_index_map
from msrjd.core.vertices import (
    extract_vertex_types, extract_source_types, SourceType, VertexType,
)
from pipeline.theory import TemporalTheoryBuilder

av, bv = 0.05, 0.05
model = (TemporalTheoryBuilder('ou-quadcubic-k3-wickcount')
         .physical_field('x')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('a', default=av).parameter('b', default=bv)
         .parameter('T', default=1.0, domain='positive')
         .set_action_text('xt*((Dt+mu)*x + a*x^2 + b*x^3) - T*xt^2')
         .equation(lhs='(Dt+mu)*x + a*x^2 + b*x^3', rhs='0').build())
ft = FieldTheory(model, taylor_order=5)
ft.expand()
prop = build_propagator(ft, model, use_cache=False, verbose=False)
num_params = {SR.var('mu'): 1.0, SR.var('a'): av, SR.var('b'): bv,
              SR.var('T'): 1.0, SR.var('xstar1'): 0.0}
compute_poles_and_residues(prop, num_params, verbose=False)
ring = list(ft._ns._ring_var_names)
resp_idx, phys_idx = build_field_index_map(ring, ft._n_tilde)
ext = [('dx', 1)] * 3
vtypes = extract_vertex_types(ft)
stypes = extract_source_types(ft)
ub, mb, _ = enumerate_unique_diagrams(ft, model, k=3, max_ell=1,
                                      external_fields=ext, G_ft=prop['G_ft'],
                                      resp_idx=resp_idx, phys_idx=phys_idx,
                                      vtypes=vtypes, stypes=stypes,
                                      use_cache=False, parallel=False,
                                      verbose=False)

asym = SR.var('a')


def td_class_key(td):
    """Collapsed-digraph canonical key with the SAME coloring."""
    leaf_list = list(td.prediagram[2])
    leaf_set = set(leaf_list)
    namemap = {}
    ai = ni = ei = 0
    for v, x in sorted(td.vertex_assignments.items(), key=lambda kv: str(kv[0])):
        if isinstance(x, VertexType):
            namemap[v] = ('A', ai); ai += 1
        elif isinstance(x, SourceType):
            namemap[v] = ('N', ni); ni += 1
    for lf in leaf_list:
        namemap[lf] = ('E', ei); ei += 1
    if ai != 3 or ni != 3 or ei != 3:
        return None     # not the a^3 sector shape
    edges = tuple((namemap[ek[0]], namemap[ek[1]]) for ek in td.edge_types)
    return canonical_key(edges)


pipe_w = defaultdict(float)
pipe_detail = defaultdict(list)
for td, mult in zip(ub.get(1, []), mb.get(1, [1] * len(ub.get(1, [])))):
    info = classify_coefficient_factors(
        td, [], {'temporal_type': 'white', 'amplitude_params': []})
    pref = SR(info['scalar_prefactor'])
    if abs(float(pref.subs(num_params))) < 1e-14:
        continue
    if pref.degree(asym) != 3:
        continue
    key = td_class_key(td)
    S = combinatorial_factor(td)
    comp = external_wick_compensation(td)
    w = S * factorial(3) / comp
    pipe_w[key] += w
    pipe_detail[key].append((S, comp, mult))

# ── 3. Compare ──────────────────────────────────────────────────────
all_keys = sorted(set(wick_w) | set(pipe_w))
print()
print('%-12s %12s %12s %8s   detail (S, comp, mult)' %
      ('class', 'wick_w', 'pipe_w', 'ratio'))
bad = 0
for k in all_keys:
    ww = wick_w.get(k, 0.0)
    pw = pipe_w.get(k, 0.0)
    ratio = (pw / ww) if ww else float('inf')
    tag = '' if abs(ratio - 1) < 1e-12 else '   <-- MISMATCH'
    if tag:
        bad += 1
    print('%-12s %12.1f %12.1f %8.4f%s   %s' %
          (k[:12], ww, pw, ratio, tag, pipe_detail.get(k, '')))
print()
print('total wick_w  = %.1f' % sum(wick_w.values()))
print('total pipe_w  = %.1f' % sum(pipe_w.values()))
print('mismatched classes: %d' % bad)

print()
print('=== representatives of MISSING classes (pipe_w == 0) ===')
for k in all_keys:
    if pipe_w.get(k, 0.0) == 0.0:
        rep = REPS[k]
        pretty = ' '.join('%s%d>%s%d' % (u[0], u[1], v[0], v[1])
                          for u, v in sorted(rep, key=str))
        print('wick_w=%6.1f   %s' % (wick_w[k], pretty))

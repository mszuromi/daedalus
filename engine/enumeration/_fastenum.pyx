# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
engine.enumeration._fastenum  (Cython)
======================================
Hot loops of the prediagram enumeration, compiled.  Loaded through
``engine.enumeration.fastenum`` (pyximport with a pure-Python fallback), so the
package still runs without a C compiler; ``DAEDALUS_FASTENUM=0`` disables it.

Three things live here.

``tree_candidates(n, eu, ev, retired, ell, cotree)``
    The whole (E2) stage for one tree and one retired-leaf subset: endpoint
    vectors with the owed-endpoint pruning of ``_degree_vectors``, their
    realizations as edge multisets (``_realizations``), the exchange-maximal
    co-tree filter (``_cotree_is_local_max``), the two structural checks of
    ``emit``, and the certificate of each survivor.

``aux_cert(n, edges, directed)``
    Isomorphism certificate of a (di)multigraph on ``n`` vertices, in the same
    ``(order, sorted_edges)`` format as ``loop_diagram_enumeration._iso_cert``.
    The multigraph is encoded as a two-coloured simple (di)graph with one
    subdivision vertex per edge copy (colour 1) between its endpoints
    (colour 0); Sage's ``search_tree`` canonically labels that graph on a
    ``DenseGraph`` directly, without the ``Graph``-object round trip that made
    ``canonical_label`` cost ~100 us.  The certificate is the multigraph
    relabelled by the canonical order of its colour-0 vertices, which is an
    isomorphism invariant because the coloured encoding is.  NOTE: the bytes
    differ from the Sage backend's for the same class; compare certificate
    sets across backends only through ``prediagram_cache.recanonicalize``.

``orientation_patterns(n, eu, ev, base, free, leaf, deg)``
    All bit patterns over the free edges for which the orientation obeys the
    four causal rules and is acyclic -- the search of
    ``enumerate_orientations_integer`` with the per-pattern work in C.
"""
from sage.groups.perm_gps.partn_ref.refinement_graphs import search_tree
from sage.graphs.base.dense_graph import DenseGraph

from libc.stdlib cimport malloc, free


def aux_cert(int n, edges, bint directed):
    cdef int m = len(edges)
    cdef int N = n + m
    cdef int i, u, v
    D = DenseGraph(N)
    for i in range(m):
        u = edges[i][0]; v = edges[i][1]
        D.add_arc(u, n + i)
        D.add_arc(n + i, v)
        if not directed:
            D.add_arc(n + i, u)
            D.add_arc(v, n + i)
    part = [list(range(n)), list(range(n, N))]
    res = search_tree(D, part, lab=True, dig=directed, certificate=True)
    cert = res[2]                                   # vertex -> canonical label
    order = sorted(range(n), key=cert.__getitem__)  # colour-0 vertices in canonical order
    cdef int* pos = <int*> malloc(n * sizeof(int))
    for i in range(n):
        pos[order[i]] = i
    out = []
    for i in range(m):
        u = pos[edges[i][0]]; v = pos[edges[i][1]]
        if directed or u <= v:
            out.append((u, v))
        else:
            out.append((v, u))
    free(pos)
    out.sort()
    return (n, tuple(out))


def orientation_patterns(int n, eu, ev, base, free_idx, leaf, deg):
    """``eu[i], ev[i]``: endpoints of edge copy ``i`` (m edges); ``base[i]``:
    forced bit (0: eu->ev, 1: ev->eu) or 0 for free edges; ``free_idx``: indices of
    the free edges; ``leaf[v]``: 1 if v has degree 1; ``deg[v]``: degree.
    Returns the list of full bit patterns (as Python ints, bit i = edge i) that
    pass the checks of ``check_orientation_constraints``."""
    cdef int m = len(eu)
    cdef int nf = len(free_idx)
    cdef int i, j, a, b, v, w, popped, ok
    cdef unsigned long long bits, pat, nfree_max
    cdef int* U = <int*> malloc(m * sizeof(int))
    cdef int* V = <int*> malloc(m * sizeof(int))
    cdef int* B = <int*> malloc(m * sizeof(int))
    cdef int* FR = <int*> malloc((nf if nf > 0 else 1) * sizeof(int))
    cdef int* indeg = <int*> malloc(n * sizeof(int))
    cdef int* outdeg = <int*> malloc(n * sizeof(int))
    cdef int* rem = <int*> malloc(n * sizeof(int))
    cdef int* stack = <int*> malloc(n * sizeof(int))
    cdef int* head = <int*> malloc(m * sizeof(int))       # oriented head of edge i
    cdef int* tail = <int*> malloc(m * sizeof(int))
    cdef int* nbr_ptr = <int*> malloc((n + 1) * sizeof(int))   # CSR undirected adjacency
    cdef int* nbr = <int*> malloc(2 * m * sizeof(int))
    cdef int* cnt = <int*> malloc(n * sizeof(int))
    cdef int* isleaf = <int*> malloc(n * sizeof(int))
    for i in range(m):
        U[i] = eu[i]; V[i] = ev[i]; B[i] = base[i]
    for i in range(nf):
        FR[i] = free_idx[i]
    for v in range(n):
        isleaf[v] = leaf[v]; cnt[v] = 0
    for i in range(m):
        cnt[U[i]] += 1; cnt[V[i]] += 1
    nbr_ptr[0] = 0
    for v in range(n):
        nbr_ptr[v + 1] = nbr_ptr[v] + cnt[v]; cnt[v] = 0
    for i in range(m):
        nbr[nbr_ptr[U[i]] + cnt[U[i]]] = V[i]; cnt[U[i]] += 1
        nbr[nbr_ptr[V[i]] + cnt[V[i]]] = U[i]; cnt[V[i]] += 1
    out = []
    nfree_max = (<unsigned long long>1) << nf
    bits = 0
    while bits < nfree_max:
        # assemble the pattern
        pat = 0
        for i in range(m):
            if B[i]:
                pat |= (<unsigned long long>1) << i
        for j in range(nf):
            if (bits >> j) & 1:
                pat |= (<unsigned long long>1) << FR[j]
        for v in range(n):
            indeg[v] = 0; outdeg[v] = 0
        for i in range(m):
            if (pat >> i) & 1:
                tail[i] = V[i]; head[i] = U[i]
            else:
                tail[i] = U[i]; head[i] = V[i]
            outdeg[tail[i]] += 1; indeg[head[i]] += 1
        ok = 1
        for v in range(n):
            a = indeg[v]; b = outdeg[v]
            if (a + b == 1 and a != 1) or (b == 0 and a >= 2) or (a == 1 and b == 1):
                ok = 0; break
        if ok:
            for v in range(n):                      # no adjacent sources
                if indeg[v] == 0:
                    for j in range(nbr_ptr[v], nbr_ptr[v + 1]):
                        if indeg[nbr[j]] == 0:
                            ok = 0; break
                    if not ok: break
        if ok:                                      # acyclic (Kahn)
            popped = 0; j = 0
            for v in range(n):
                rem[v] = indeg[v]
                if rem[v] == 0:
                    stack[j] = v; j += 1
            while j > 0:
                j -= 1; v = stack[j]; popped += 1
                for i in range(m):
                    if tail[i] == v:
                        w = head[i]; rem[w] -= 1
                        if rem[w] == 0:
                            stack[j] = w; j += 1
            if popped != n:
                ok = 0
        if ok:
            out.append(pat)
        bits += 1
    free(U); free(V); free(B); free(FR); free(indeg); free(outdeg); free(rem); free(stack)
    free(head); free(tail); free(nbr_ptr); free(nbr); free(cnt); free(isleaf)
    return out


# ---------------------------------------------------------------------------
# (E2) for one tree and one retired-leaf subset, entirely in C: endpoint
# vectors (with the owed-endpoint and adjacency pruning of
# ``_degree_vectors``), their realizations as edge multisets
# (``_realizations``), the exchange-maximal co-tree filter
# (``_cotree_is_local_max``) and the two structural checks of ``emit``; each
# survivor is certified with ``aux_cert``.  Same set of certificates as the
# Python path (tests/test_fastenum.py).
# ---------------------------------------------------------------------------

cdef struct TS:
    int n, L, ell, cotree
    int* adj_ptr
    int* adj
    int* eu
    int* ev
    int* base_deg
    int* is_d2
    int* is_retired
    int* order
    int* delta
    int* parent          # parent[r*n + v] on the tree rooted at r


cdef inline int in_d2(TS* s, int u) noexcept:
    cdef int d = s.delta[u]
    if d < 0:
        return 0
    return (s.is_d2[u] and d == 0) or (s.is_retired[u] and d == 1)


cdef inline int any_nbr_in_d2(TS* s, int v) noexcept:
    cdef int j
    for j in range(s.adj_ptr[v], s.adj_ptr[v + 1]):
        if in_d2(s, s.adj[j]):
            return 1
    return 0


cdef int owed(TS* s, int i) noexcept:
    cdef int need = 0, idx, v
    for idx in range(i, s.L):
        v = s.order[idx]
        if s.is_retired[v]:
            need += 1
        elif s.is_d2[v] and any_nbr_in_d2(s, v):
            need += 1
    return need


cdef inline int forbidden(TS* s, int u, int w) noexcept:
    return s.is_retired[u] and s.delta[u] == 1 and s.is_retired[w] and s.delta[w] == 1


cdef void on_F(TS* s, int* F, int nF, list out):
    cdef int n = s.n, i, u, w, x, p, v, j, a, b
    cdef int rf_hi, rf_lo, re_hi, re_lo, du, dw
    cdef int* deg = <int*> malloc(n * sizeof(int))
    for v in range(n):
        deg[v] = s.base_deg[v] + (s.delta[v] if s.delta[v] > 0 else 0)
    if s.cotree:
        for i in range(nF):
            u = F[2 * i]; w = F[2 * i + 1]
            du = deg[u]; dw = deg[w]
            if du >= dw:
                rf_hi = du; rf_lo = dw
            else:
                rf_hi = dw; rf_lo = du
            x = w
            while x != u:
                p = s.parent[u * n + x]
                a = deg[x]; b = deg[p]
                if a >= b:
                    re_hi = a; re_lo = b
                else:
                    re_hi = b; re_lo = a
                if re_hi > rf_hi or (re_hi == rf_hi and re_lo > rf_lo):
                    free(deg)
                    return
                x = p
    # structural checks (only for ell > 0, which is the only case handled here)
    cdef int* is2 = <int*> malloc(n * sizeof(int))
    for v in range(n):
        is2[v] = 1 if deg[v] == 2 else 0
    cdef int alld2, has_ln = 0, ln_alld2 = 1
    for v in range(n):
        if deg[v] >= 3:
            alld2 = 1
            for j in range(s.adj_ptr[v], s.adj_ptr[v + 1]):
                if not is2[s.adj[j]]:
                    alld2 = 0; break
            if alld2:
                for i in range(nF):
                    if F[2 * i] == v and not is2[F[2 * i + 1]]:
                        alld2 = 0; break
                    if F[2 * i + 1] == v and not is2[F[2 * i]]:
                        alld2 = 0; break
            if alld2:
                free(deg); free(is2)
                return
    for v in range(n):
        if deg[v] == 1:                                   # a leaf; its neighbours
            for j in range(s.adj_ptr[v], s.adj_ptr[v + 1]):
                has_ln = 1
                if not is2[s.adj[j]]:
                    ln_alld2 = 0
            for i in range(nF):
                if F[2 * i] == v:
                    has_ln = 1
                    if not is2[F[2 * i + 1]]:
                        ln_alld2 = 0
                if F[2 * i + 1] == v:
                    has_ln = 1
                    if not is2[F[2 * i]]:
                        ln_alld2 = 0
    if has_ln and ln_alld2:
        free(deg); free(is2)
        return
    free(deg); free(is2)
    edges = []
    for i in range(n - 1):
        edges.append((s.eu[i], s.ev[i]))
    for i in range(nF):
        edges.append((F[2 * i], F[2 * i + 1]))
    out.append(aux_cert(n, edges, False))


cdef void rec_real(TS* s, int* rem, int* F, int nF, list out):
    cdef int u = -1, v, need
    for v in range(s.n):
        if rem[v] > 0:
            u = v; break
    if u == -1:
        on_F(s, F, nF, out)
        return
    need = rem[u]
    rem[u] = 0
    choose(s, rem, F, nF, u, need, u + 1, out)
    rem[u] = need


cdef void choose(TS* s, int* rem, int* F, int nF, int u, int left, int start, list out):
    cdef int w
    if left == 0:
        rec_real(s, rem, F, nF, out)
        return
    for w in range(start, s.n):
        if rem[w] <= 0:
            continue
        if forbidden(s, u, w):
            continue
        rem[w] -= 1
        F[2 * nF] = u; F[2 * nF + 1] = w
        choose(s, rem, F, nF + 1, u, left - 1, w, out)
        rem[w] += 1


cdef void on_delta(TS* s, list out):
    cdef int v
    cdef int* rem = <int*> malloc(s.n * sizeof(int))
    cdef int* F = <int*> malloc(2 * (s.ell + 1) * sizeof(int))
    for v in range(s.n):
        rem[v] = s.delta[v] if s.delta[v] > 0 else 0
    rec_real(s, rem, F, 0, out)
    free(rem); free(F)


cdef void rec_delta(TS* s, int i, int rem, list out):
    cdef int v, lo, d
    if rem < owed(s, i):
        return
    if i == s.L:
        if rem == 0:
            on_delta(s, out)
        return
    v = s.order[i]
    lo = 1 if s.is_retired[v] else 0
    for d in range(lo, rem + 1):
        if (s.is_d2[v] and d == 0) or (s.is_retired[v] and d == 1):
            if any_nbr_in_d2(s, v):
                continue
        s.delta[v] = d
        rec_delta(s, i + 1, rem - d, out)
        s.delta[v] = -1


def tree_candidates(int n, eu, ev, retired, int ell, bint cotree):
    """Certificates of the topologies obtained from the tree ``(n, eu, ev)``
    (n-1 edges, vertices 0..n-1) by adding ``ell`` edges that retire exactly
    the leaves in ``retired`` and pass the structural checks (and, with
    ``cotree``, the exchange-maximal filter).  ``ell >= 1``."""
    cdef TS s
    cdef int i, v, j, u, w, r, top
    cdef int m = n - 1
    s.n = n; s.ell = ell; s.cotree = 1 if cotree else 0
    s.eu = <int*> malloc(m * sizeof(int)); s.ev = <int*> malloc(m * sizeof(int))
    s.base_deg = <int*> malloc(n * sizeof(int)); s.is_d2 = <int*> malloc(n * sizeof(int))
    s.is_retired = <int*> malloc(n * sizeof(int)); s.delta = <int*> malloc(n * sizeof(int))
    s.adj_ptr = <int*> malloc((n + 1) * sizeof(int)); s.adj = <int*> malloc(2 * m * sizeof(int))
    s.order = <int*> malloc(n * sizeof(int)); s.parent = <int*> malloc(n * n * sizeof(int))
    cdef int* cnt = <int*> malloc(n * sizeof(int))
    cdef int* stack = <int*> malloc(n * sizeof(int))
    for v in range(n):
        s.base_deg[v] = 0; s.is_retired[v] = 0; s.delta[v] = -1; cnt[v] = 0
    for i in range(m):
        s.eu[i] = eu[i]; s.ev[i] = ev[i]
        s.base_deg[s.eu[i]] += 1; s.base_deg[s.ev[i]] += 1
    s.adj_ptr[0] = 0
    for v in range(n):
        s.adj_ptr[v + 1] = s.adj_ptr[v] + s.base_deg[v]
    for i in range(m):
        u = s.eu[i]; w = s.ev[i]
        s.adj[s.adj_ptr[u] + cnt[u]] = w; cnt[u] += 1
        s.adj[s.adj_ptr[w] + cnt[w]] = u; cnt[w] += 1
    for v in range(n):
        s.is_d2[v] = 1 if s.base_deg[v] == 2 else 0
    for v in retired:
        s.is_retired[v] = 1
    # order: retired leaves (given order), then tree degree-2 vertices, then other internal vertices
    s.L = 0
    for v in retired:
        s.order[s.L] = v; s.L += 1
    for v in range(n):
        if s.is_d2[v]:
            s.order[s.L] = v; s.L += 1
    for v in range(n):
        if s.base_deg[v] >= 3:
            s.order[s.L] = v; s.L += 1
    # parent tables
    for r in range(n):
        for v in range(n):
            s.parent[r * n + v] = -2
        s.parent[r * n + r] = -1
        top = 0; stack[top] = r; top += 1
        while top > 0:
            top -= 1; v = stack[top]
            for j in range(s.adj_ptr[v], s.adj_ptr[v + 1]):
                w = s.adj[j]
                if s.parent[r * n + w] == -2:
                    s.parent[r * n + w] = v
                    stack[top] = w; top += 1
    out = []
    rec_delta(&s, 0, 2 * ell, out)
    free(s.eu); free(s.ev); free(s.base_deg); free(s.is_d2); free(s.is_retired); free(s.delta)
    free(s.adj_ptr); free(s.adj); free(s.order); free(s.parent); free(cnt); free(stack)
    return out

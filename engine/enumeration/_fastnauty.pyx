# cython: language_level=3, boundscheck=False, wraparound=False
"""
engine.enumeration._fastnauty  (Cython, links Sage's bundled libnauty)
=====================================================================
``nauty_cert(n, edges, directed)``: isomorphism certificate of a (di)multigraph
in the ``(order, sorted_edges)`` format of ``loop_diagram_enumeration._iso_cert``,
computed by nauty's ``densenauty`` on the same two-coloured subdivision
encoding as ``_fastenum.aux_cert`` (one colour-1 vertex per edge copy; for directed graphs a colour-1 and a colour-2 vertex in series, so nauty runs undirected).  The
certificate is the multigraph relabelled by the canonical order of its
colour-0 vertices.  Bytes differ from both other backends for the same class;
compare across backends only through ``prediagram_cache.recanonicalize``.
"""
from libc.stdlib cimport malloc, free

cdef extern from *:
    """
    #include <stdlib.h>
    #include <string.h>
    #include "nauty.h"

    /* canonical index of every vertex of the subdivision encoding */
    /* canonical index of every vertex of the coloured subdivision encoding.
       undirected: u - a - v with a of colour 1 (N = n + ne)
       directed:   u - a - b - v with a of colour 1, b of colour 2 (N = n + 2 ne);
       the two colours record the direction, so nauty runs in its (much
       faster) undirected mode either way. */
    static int fe_nauty_canon(int n, const int *eu, const int *ev, int ne,
                              int directed, int *canon_index)
    {
        int N = directed ? n + 2 * ne : n + ne, m = SETWORDSNEEDED(N), i, u, v, a, b;
        graph *g, *cg; int *lab, *ptn, *orbits;
        DEFAULTOPTIONS_GRAPH(opts);
        statsblk stats;
        nauty_check(WORDSIZE, m, N, NAUTYVERSIONID);
        g  = (graph*) calloc((size_t)N * m, sizeof(graph));
        cg = (graph*) calloc((size_t)N * m, sizeof(graph));
        lab = (int*) malloc(N * sizeof(int)); ptn = (int*) malloc(N * sizeof(int));
        orbits = (int*) malloc(N * sizeof(int));
        if (!g || !cg || !lab || !ptn || !orbits) return -1;
        for (i = 0; i < ne; ++i) {
            u = eu[i]; v = ev[i]; a = n + i;
            if (directed) {
                b = n + ne + i;
                ADDONEEDGE(g, u, a, m); ADDONEEDGE(g, a, b, m); ADDONEEDGE(g, b, v, m);
            } else {
                ADDONEEDGE(g, u, a, m); ADDONEEDGE(g, a, v, m);
            }
        }
        for (i = 0; i < N; ++i) { lab[i] = i; ptn[i] = 1; }
        if (n > 0) ptn[n - 1] = 0;
        if (directed && ne > 0) ptn[n + ne - 1] = 0;
        ptn[N - 1] = 0;
        opts.getcanon = TRUE; opts.defaultptn = FALSE;
        densenauty(g, lab, ptn, orbits, &opts, &stats, m, N, cg);
        for (i = 0; i < N; ++i) canon_index[lab[i]] = i;
        free(g); free(cg); free(lab); free(ptn); free(orbits);
        return 0;
    }

    /* general coloured version: colour[i] for all N = n + 2 ne encoded vertices,
       cells ordered by colour value; canonical index of every vertex */
    static int fe_nauty_canon_coloured(int n, const int *eu, const int *ev, int ne,
                                       const int *colour, int *canon_index)
    {
        int N = n + 2 * ne, m = SETWORDSNEEDED(N), i, u, v, a, b, c, maxc = 0, k;
        graph *g, *cg; int *lab, *ptn, *orbits, *cnt;
        DEFAULTOPTIONS_GRAPH(opts);
        statsblk stats;
        nauty_check(WORDSIZE, m, N, NAUTYVERSIONID);
        g  = (graph*) calloc((size_t)N * m, sizeof(graph));
        cg = (graph*) calloc((size_t)N * m, sizeof(graph));
        lab = (int*) malloc(N * sizeof(int)); ptn = (int*) malloc(N * sizeof(int));
        orbits = (int*) malloc(N * sizeof(int));
        if (!g || !cg || !lab || !ptn || !orbits) return -1;
        for (i = 0; i < ne; ++i) {
            u = eu[i]; v = ev[i]; a = n + i; b = n + ne + i;
            ADDONEEDGE(g, u, a, m); ADDONEEDGE(g, a, b, m); ADDONEEDGE(g, b, v, m);
        }
        for (i = 0; i < N; ++i) if (colour[i] > maxc) maxc = colour[i];
        cnt = (int*) calloc((size_t)maxc + 2, sizeof(int));
        if (!cnt) return -1;
        for (i = 0; i < N; ++i) cnt[colour[i] + 1]++;
        for (c = 0; c <= maxc; ++c) cnt[c + 1] += cnt[c];        /* prefix sums: cell starts */
        for (i = 0; i < N; ++i) { lab[cnt[colour[i]]++] = i; }  /* stable bucket sort by colour */
        for (i = 0; i < N; ++i) ptn[i] = 1;
        k = 0;
        for (c = 0; c <= maxc; ++c) {                            /* cell ends: cnt[c] now = end of cell c */
            if (cnt[c] > 0 && cnt[c] > k) { ptn[cnt[c] - 1] = 0; k = cnt[c]; }
        }
        ptn[N - 1] = 0;
        opts.getcanon = TRUE; opts.defaultptn = FALSE;
        densenauty(g, lab, ptn, orbits, &opts, &stats, m, N, cg);
        for (i = 0; i < N; ++i) canon_index[lab[i]] = i;
        free(g); free(cg); free(lab); free(ptn); free(orbits); free(cnt);
        return 0;
    }
    """
    int fe_nauty_canon(int n, const int* eu, const int* ev, int ne, int directed, int* canon_index) nogil
    int fe_nauty_canon_coloured(int n, const int* eu, const int* ev, int ne, const int* colour, int* canon_index) nogil


def nauty_cert(int n, edges, bint directed):
    cdef int m = len(edges)
    cdef int N = n + 2 * m
    cdef int i, u, v, rc
    cdef int* eu = <int*> malloc((m if m > 0 else 1) * sizeof(int))
    cdef int* ev = <int*> malloc((m if m > 0 else 1) * sizeof(int))
    cdef int* ci = <int*> malloc((N if N > 0 else 1) * sizeof(int))
    cdef int* pos = <int*> malloc((n if n > 0 else 1) * sizeof(int))
    for i in range(m):
        eu[i] = edges[i][0]; ev[i] = edges[i][1]
    rc = fe_nauty_canon(n, eu, ev, m, 1 if directed else 0, ci)
    if rc != 0:
        free(eu); free(ev); free(ci); free(pos)
        raise MemoryError('nauty_cert')
    # colour-0 vertices in canonical order -> positions 0..n-1
    order = sorted(range(n), key=lambda x: ci[x])
    for i in range(n):
        pos[order[i]] = i
    out = []
    for i in range(m):
        u = pos[eu[i]]; v = pos[ev[i]]
        if directed or u <= v:
            out.append((u, v))
        else:
            out.append((v, u))
    out.sort()
    free(eu); free(ev); free(ci); free(pos)
    return (n, tuple(out))


def nauty_cert_coloured(int n, edges, vcol, ecol):
    """Certificate of a directed multigraph with vertex colours ``vcol[v]``
    and edge-copy colours ``ecol[i]`` (small non-negative ints), complete up
    to colour-preserving isomorphism.  Encoding: vertex v keeps its colour;
    edge copy i = (u -> v) becomes u - a_i - b_i - v with a_i coloured by
    ``ecol[i]`` (offset past the vertex colours) and all b_i in one further
    colour, so direction and edge colour are both recorded and nauty runs
    undirected.  Returns ``(n, vertex_colours_in_canonical_order,
    sorted_coloured_edges)``."""
    cdef int m = len(edges)
    cdef int N = n + 2 * m
    cdef int i, u, v, rc, nvc, nec
    nvc = (max(vcol) + 1) if n > 0 else 0
    nec = (max(ecol) + 1) if m > 0 else 0
    cdef int* eu = <int*> malloc((m if m > 0 else 1) * sizeof(int))
    cdef int* ev = <int*> malloc((m if m > 0 else 1) * sizeof(int))
    cdef int* colour = <int*> malloc((N if N > 0 else 1) * sizeof(int))
    cdef int* ci = <int*> malloc((N if N > 0 else 1) * sizeof(int))
    cdef int* pos = <int*> malloc((n if n > 0 else 1) * sizeof(int))
    for i in range(n):
        colour[i] = vcol[i]
    for i in range(m):
        eu[i] = edges[i][0]; ev[i] = edges[i][1]
        colour[n + i] = nvc + ecol[i]
        colour[n + m + i] = nvc + nec
    rc = fe_nauty_canon_coloured(n, eu, ev, m, colour, ci)
    if rc != 0:
        free(eu); free(ev); free(colour); free(ci); free(pos)
        raise MemoryError('nauty_cert_coloured')
    order = sorted(range(n), key=lambda x: ci[x])
    for i in range(n):
        pos[order[i]] = i
    vc = tuple(vcol[order[i]] for i in range(n))
    out = []
    for i in range(m):
        out.append((pos[eu[i]], pos[ev[i]], ecol[i]))
    out.sort()
    free(eu); free(ev); free(colour); free(ci); free(pos)
    return (n, vc, tuple(out))

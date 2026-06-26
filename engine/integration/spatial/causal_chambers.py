"""
engine.integration.spatial.causal_chambers
==========================================
Backend C — **C2-full core**: the causal ordering chambers of a loop diagram's
internal-vertex time integral, with **smooth per-chamber quadrature**.

This is the multi-vertex generalization of ``temporal_integrate.sigma_parametric``
(which handles only the 2-vertex, single-ordering case — the bubble and sunset).
A general loop self-energy has internal vertices whose times are integrated, and
the retarded ``θ``'s carve that time space into **ordering chambers**.

**Reuse.** The poset construction + linear-extension (chamber) enumeration are
*representation-independent* — they act on the vertex-time orderings — so we reuse
the temporal pipeline's battle-tested machinery verbatim:
``final_integral._CausalPoset`` and ``_enumerate_linear_extensions``.

**The one difference (the whole point of momentum-first).** The temporal pipeline
integrates *products of exponentials* per chamber (``_exp_over_chain_simplex``),
whose ``1/β`` factors are the close-pair pathology.  Here the loop momenta are
already integrated (the C0/C1 Symanzik step), so the per-chamber integrand is
**smooth** — we use ordinary quadrature and **close-pair cannot arise**.  (As a
bonus, the integrand is smooth *within* each chamber even when it contains
``|Δt|`` correlation factors, because a fixed ordering resolves every ``|t_i−t_j|``
into a definite sign — the only kinks are exactly the chamber boundaries.)

SCOPE (this module): the chamber decomposition + smooth simplex quadrature over
the internal vertex times.  Extracting a specific enumerated diagram's
retarded/correlation edge structure and Symanzik integrand (C0/C1) and threading
amputation / external-τ through this is the next step.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate

from engine.integration.time_domain.final_integral import (
    _CausalPoset, _enumerate_linear_extensions,
)


def causal_chambers(n_vertices, retarded_edges):
    """The ordering chambers of the internal-vertex-time poset — **reuses** the
    temporal ``_CausalPoset`` + ``_enumerate_linear_extensions``.

    ``retarded_edges`` : iterable of ``(u, v)`` meaning a retarded line ``u→v``,
    i.e. the causal ordering ``t_v > t_u``.  Returns the list of linear
    extensions, each a length-``n_vertices`` tuple giving the vertex order
    (earliest vertex first).  An empty edge set ⇒ all ``n!`` orderings (which
    tile the time cube).
    """
    poset = _CausalPoset(
        m=int(n_vertices),
        edges=tuple(sorted({(int(u), int(v)) for (u, v) in retarded_edges})),
        scalar_lowers=(), scalar_uppers=())
    return list(_enumerate_linear_extensions(poset))


def integrate_chamber(f, order, lo, hi, limit=80):
    """``∫ f(t) dt`` over the chamber simplex
    ``{lo ≤ t_{order[0]} ≤ … ≤ t_{order[-1]} ≤ hi}`` by nested 1-D quadrature.

    ``f`` takes a length-``n`` array of vertex times (indexed by **vertex**, not
    by order) and must be smooth on the simplex interior (the C0/C1 momentum
    integral guarantees this — no poles).  ``order`` is one chamber from
    :func:`causal_chambers`.
    """
    n = len(order)
    tv = np.zeros(n)

    def rec(level, upper):
        if level < 0:
            return float(f(tv))
        var = order[level]

        def g(x):
            tv[var] = x
            return rec(level - 1, x)
        val, _ = integrate.quad(g, lo, upper, limit=limit)
        return val

    return rec(n - 1, hi)


def integrate_over_chambers(f, n_vertices, retarded_edges, lo, hi, limit=80):
    """The C2-full internal-time integral: **Σ over causal chambers** of the
    smooth simplex integral.  Equals ``∫_{[lo,hi]^n} f(t)·𝟙(retarded orderings) dt``
    — the chambers partition the constrained domain, and within each the
    integrand is smooth.
    """
    return float(sum(
        integrate_chamber(f, order, lo, hi, limit=limit)
        for order in causal_chambers(n_vertices, retarded_edges)))

"""
tests/test_causal_chambers.py
=============================
Backend C — C2-full core (``msrjd.integration.spatial.causal_chambers``): the
reused causal-poset chamber enumeration + smooth per-chamber quadrature.

Validates:
  * chamber ENUMERATION (reused temporal `_CausalPoset`/`_enumerate_linear_extensions`):
    counts + every chamber respects the retarded edges;
  * the TILING identity: Σ over all n! chambers of the simplex integral equals
    the full-cube integral (smooth integrand) — i.e. the chambers partition the
    domain and the per-chamber quadrature is correct;
  * the EDGE-restricted identity: Σ over the chambers allowed by a retarded edge
    equals an INDEPENDENT direct quadrature over {t_v > t_u}.

Run:  sage -python -m pytest tests/test_causal_chambers.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from msrjd.integration.spatial.causal_chambers import (
    causal_chambers, integrate_over_chambers,
)

LO, HI = -7.0, 7.0


def _respects(order, edges):
    pos = {v: i for i, v in enumerate(order)}
    return all(pos[u] < pos[v] for (u, v) in edges)


# ── chamber enumeration (the reused poset machinery) ──────────────
def test_chamber_counts_and_validity():
    assert len(causal_chambers(2, [])) == 2
    assert len(causal_chambers(3, [])) == 6                # 3! orderings
    c1 = causal_chambers(3, [(0, 1)])
    assert len(c1) == 3 and all(_respects(o, [(0, 1)]) for o in c1)
    c2 = causal_chambers(3, [(0, 1), (1, 2)])
    assert c2 == [(0, 1, 2)]                                # total order
    c3 = causal_chambers(4, [(0, 1), (0, 2)])               # 0 first, 1&2&3 free
    assert all(_respects(o, [(0, 1), (0, 2)]) for o in c3)


# ── tiling: Σ chambers (no edges) == full cube ────────────────────
def test_chamber_sum_tiles_cube():
    # smooth-within-chamber integrand with correlation-style |Δt| kinks at the
    # chamber boundaries (mimics C(|Δt|)=e^{-m|Δt|} factors).
    def f(t):
        return math.exp(-(abs(t[0] - t[1]) + abs(t[1] - t[2])))
    got = integrate_over_chambers(f, 3, [], LO, HI)
    ref, _ = integrate.nquad(lambda a, b, c: f((a, b, c)),
                             [[LO, HI], [LO, HI], [LO, HI]])
    assert abs(got - ref) <= 1e-3 * abs(ref)


# ── edge-restricted: Σ allowed chambers == direct ∫ over {t1>t0} ──
def test_chamber_sum_edge_matches_direct():
    # one retarded edge 0→1 (t1>t0) + a correlation factor on 1–2.
    def f(t):
        return math.exp(-(t[1] - t[0]) - abs(t[1] - t[2]))
    got = integrate_over_chambers(f, 3, [(0, 1)], LO, HI)   # 3 allowed chambers
    # independent reference: tplquad with t1∈[t0,HI] (the edge as a limit).
    # tplquad(func(z,y,x)) : x=t0∈[LO,HI], y=t2∈[LO,HI], z=t1∈[t0,HI]
    ref, _ = integrate.tplquad(
        lambda t1, t2, t0: f((t0, t1, t2)),
        LO, HI,
        lambda t0: LO, lambda t0: HI,
        lambda t0, t2: t0, lambda t0, t2: HI)
    assert abs(got - ref) <= 1e-3 * abs(ref)

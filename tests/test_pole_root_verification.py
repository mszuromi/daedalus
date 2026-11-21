"""
The roots behind the analyzer's pole-based stability verdict
(``daedalus.api._propagator.verified_roots``).

A rational model's stability is read off the roots of ``Q(omega)``, the
common denominator of ``G = K^{-1}``.  A Pade approximant of a delay makes
``Q`` of degree ~ 100 and more, and a double-precision root finder then places
some of its roots far off, in the lower half plane, although every true root
is retarded: a stable delayed model was refused as unstable.  When a double
root lies on or below the real axis, all roots are recomputed from the exact
coefficients in high precision.  A truly unstable root stays, and an unstable
delayed model is still refused (by its characteristic function, before any
approximant runs).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                                      # noqa: E402
from daedalus.api.delay import Kernel                                           # noqa: E402
from daedalus.api.sde import SDE                                                # noqa: E402


def _poly(roots):
    from sage.all import CyclotomicField, PolynomialRing
    CF = CyclotomicField(4, 'I_')
    i = CF.gen()
    PR = PolynomialRing(CF, 'om')
    x = PR.gen()
    Q = PR(1)
    for re, im in roots:
        Q *= x - (re + im * i)
    return Q


def _double_roots(Q):
    from sage.all import CDF, PolynomialRing
    return [complex(r) for r, _ in
            PolynomialRing(CDF, 'z')([complex(c) for c in Q.list()]).roots(CDF)]


def test_high_degree_retarded_roots_are_verified():
    """Roots k + i/2, k = 1..30: double precision puts some below the real
    axis; the verified roots are all retarded and exact."""
    from sage.all import QQ
    from daedalus.api._propagator import verified_roots
    Q = _poly([(k, QQ(1) / 2) for k in range(1, 31)])
    dr = _double_roots(Q)
    assert any(r.imag < 0 for r in dr)          # the defect the check guards against
    vr = verified_roots(Q, dr)
    assert len(vr) == 30
    assert np.allclose(sorted(r.real for r in vr), np.arange(1, 31), atol=1e-12)
    assert np.allclose([r.imag for r in vr], 0.5, atol=1e-12)


def test_pole_record_has_no_spurious_lower_root():
    """The record the analyzer's G3 gate reads."""
    from sage.all import QQ, matrix
    from daedalus.api._propagator import _record_pole_analysis
    Q = _poly([(k, QQ(1) / 2) for k in range(1, 31)])
    G = matrix(Q.parent().fraction_field(), 1, 1, [1 / Q])
    out = {}
    _record_pole_analysis(out, G, Q, 1)
    assert out['degree'] == 30
    assert all(p.imag > 0 for p in out['poles'])


def test_a_true_lower_root_stays():
    from daedalus.api._propagator import verified_roots
    Q = _poly([(1, -2), (3, 1), (-2, 1)])
    vr = sorted(verified_roots(Q, _double_roots(Q)), key=lambda p: p.imag)
    assert abs(vr[0] - (1 - 2j)) < 1e-12
    assert all(r.imag > 0 for r in vr[1:])


def test_well_placed_roots_are_returned_unchanged():
    from daedalus.api._propagator import verified_roots
    Q = _poly([(1, 1), (-2, 3)])
    dr = _double_roots(Q)
    assert verified_roots(Q, dr) == dr


def test_unstable_two_delay_hawkes_is_refused():
    """Strong delayed inhibition through two distinct delays: det K(s) has
    roots with Re s > 0, and the run stops before any approximant."""
    from daedalus.api.analyze import AnalyzerError
    m = (SDE('unstable delayed inhibition')
         .parameter('mu', default=0.5).parameter('w1', default=-1.6).parameter('w2', default=-1.4)
         .parameter('d1', default=0.6).parameter('t1', default=0.2)
         .parameter('d2', default=1.0).parameter('t2', default=0.3)
         .define_kernel('g1', kernel=Kernel.delay('d1') * Kernel.exponential('t1'))
         .define_kernel('g2', kernel=Kernel.delay('d2') * Kernel.exponential('t2'))
         .point_process('n', 'mu + w1*Conv(g1, n) + w2*Conv(g2, n)')).build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(AnalyzerError) as ei:
            compute_cumulants(m, k=2, max_ell=0, external_fields=['n', 'n'],
                              tau_grid=np.array([0.0]), verbose=False)
    assert 'G2_UNSTABLE_SADDLE' in ei.value.codes

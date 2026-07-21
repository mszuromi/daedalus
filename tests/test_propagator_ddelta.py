"""
tests/test_propagator_ddelta.py
===============================
Regression test for the ``D_delta`` (instantaneous / δ(t) part of the
retarded propagator) per-entry-denominator bug in
``api/_propagator.py::_compute_residues_via_polynomial_fracfield``.

The bug (fixed): the ``D_delta = lim_{ω→∞} G(ω)`` block compared each
entry's numerator degree against the **global** common-denominator degree
``Q_cdf.degree()`` and divided by the global leading coefficient, instead of
the entry's **own** denominator ``Q_entry_cdf[i][j]``. When the propagator has
≥2 distinct poles the global LCD degree exceeds an individual entry's own
denominator degree, so ``deg(P) == Q_deg`` was False for every proper/constant
entry and the whole ``D_delta`` matrix collapsed to zero. That dropped the
instantaneous δ-part of the propagator and, downstream, the δ×smooth tree
cross-term — which made same-field auto-correlators unable to flip sign and
~10× too small. The fix uses the per-entry denominator (the same object the
residue loop already uses).

This test is model-free: it feeds a synthetic ``K_ft`` straight to the fixed
helper, so it isolates the root cause and cannot be masked by any downstream
diagram-classification logic.

Run:
    sage -python -m pytest tests/test_propagator_ddelta.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sage.all import SR, I, matrix

from api._propagator import _compute_residues_via_polynomial_fracfield


def _synthetic_two_pole_propagator():
    """Return (K_ft, omega) whose inverse G has TWO DISTINCT poles plus:

      * a const+pole entry G[0,0] = 2/5 + (1/10)/d1 whose OWN denominator is a
        single pole factor d1 — a PROPER divisor of the 2-pole global LCD
        d1·d2 (this is the entry the bug silently zeroed; lim = 0.4), and
      * a bare constant entry G[1,1] = 1 (own denominator degree 0; lim = 1.0).

    This is the minimal structure that triggers the global-vs-per-entry bug:
    an entry whose own-denominator degree is strictly less than the global LCD
    degree.
    """
    omega = SR.var('omega')
    d1 = I * omega + 1          # retarded pole at ω = i
    d2 = I * omega + 2          # retarded pole at ω = 2i   (distinct from d1)
    G = matrix(SR, [
        [SR('2/5') + SR('1/10') / d1, SR('1/5') / d2],
        [SR('3/10') / d1,             SR(1)],
    ])
    return G.inverse(), omega


def test_ddelta_uses_per_entry_denominator_multipole():
    K, omega = _synthetic_two_pole_propagator()

    pole_vals, C_mats, D_delta = _compute_residues_via_polynomial_fracfield(
        K, omega, {}, 2, verbose=False)

    Dd = np.array([[complex(D_delta[i, j]) for j in range(2)]
                   for i in range(2)])

    # Precondition for the bug: two DISTINCT retarded poles are present, so the
    # global LCD (d1·d2) has higher degree than the individual entries.
    assert len(pole_vals) == 2

    # Primary regression guard — FAILS on the buggy code, which zeroed the
    # entire matrix.
    assert np.any(np.abs(Dd) > 1e-9), (
        "D_delta collapsed to all-zero — the global-vs-per-entry LCD bug in "
        "_compute_residues_via_polynomial_fracfield has regressed."
    )

    # The two instantaneous parts: a const+pole entry (own denom = a proper
    # divisor of the LCD) and a bare constant entry.
    assert abs(Dd[0, 0] - 0.4) < 1e-6
    assert abs(Dd[1, 1] - 1.0) < 1e-6
    # Strictly-proper entries vanish at ω → ∞.
    assert abs(Dd[0, 1]) < 1e-9
    assert abs(Dd[1, 0]) < 1e-9

    # Model-agnostic invariant: D_delta == lim_{ω→∞} K_ft^{-1}, cross-checked
    # against a direct large-|ω| numerical matrix inverse (independent of the
    # symbolic fraction-field path entirely).
    K_big = np.array([[complex(K[i, j].subs({omega: 1e10}))
                       for j in range(2)] for i in range(2)])
    oracle = np.linalg.inv(K_big)
    assert np.max(np.abs(Dd - oracle)) < 1e-6


def test_ddelta_omega_inf_limit_expands_unevaluated_products():
    """Site B regression: ``_omega_inf_limit_fast`` (the LEAN symbolic D_delta
    path in build_propagator) must ``.expand()`` its numerator/denominator before
    reading the leading coefficient.

    Sage's ``.numerator()`` can return an UNEVALUATED product (from a cofactor
    whose cross-coupling cancels multiplicatively), and ``.coefficient(omega, deg)``
    does NOT auto-distribute a product — it returns a spurious 0, so the ω→∞ limit
    comes back 0 instead of the true constant, silently dropping slave-field
    instantaneous D_delta identity entries.

    Model: ``single_population_linear_delta_spikes_test`` — nf=4, lean symbolic
    path, with an algebraic SLAVE field ``v`` (no ∂ₜ) and an exponential membrane
    Conv kernel.  Pre-fix, the two neuron-2 v-slave identity entries came back 0.
    """
    import daedalus as dd
    from sage.all import SR
    from api._propagator import build_propagator
    from api._mean_field import solve_mean_field
    from engine.core.field_theory import FieldTheory

    model, _ = dd.load_model('single_population_linear_delta_spikes_test')
    ft = FieldTheory(model, taylor_order=2); ft.expand()
    prop = build_propagator(ft, model, verbose=False, use_cache=False)
    D_delta = prop['D_delta']
    assert D_delta is not None, "lean symbolic path should populate D_delta for nf=4"
    n = D_delta.nrows()

    fund = dict(Em=[0.8, 0.78], tau=[10, 9], w=[[0.0, 0.25], [0.2, 0.0]])
    num_params = solve_mean_field(ft, model, fund, verbose=False)['num_params']
    Dd = np.array([[complex(SR(D_delta[i, j]).subs(num_params))
                    for j in range(n)] for i in range(n)])

    # The two neuron-2 slave-field (v2) instantaneous identity entries. Convention:
    # rows = response (nt1,nt2,vt1,vt2), cols = physical (dn1,dn2,dv1,dv2). These are
    # the exact cells the spurious-zero bug dropped (→ 0) on the pre-fix helper.
    assert abs(Dd[1, 3] - 1.0) < 1e-9, "<n2 v~2> instantaneous coupling dropped"
    assert abs(Dd[3, 3] - 1.0) < 1e-9, "v2 slave-field self-identity dropped"

    # Model-agnostic invariant: D_delta == lim_{ω→∞} K_ft^{-1}, cross-checked against
    # a direct large-|ω| numerical inverse (convention-independent).
    omega = prop.get('omega') or SR.var('omega')
    K_ft = prop['K_ft']
    K_big = np.array([[complex(SR(K_ft[i, j]).subs(num_params).subs({omega: 1e10}))
                       for j in range(n)] for i in range(n)])
    oracle = np.linalg.inv(K_big)
    assert np.max(np.abs(Dd - oracle)) < 1e-6

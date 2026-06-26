"""
tests/test_grouped_vs_perdiag.py
================================
End-to-end correctness check for the prototype grouped Phase J path
(``api/_grouped_phase_j.compute_correction_td_grouped``) against
the canonical per-diagram path (``compute_correction_td``).

Mathematical equivalence
------------------------
By linearity of integration, the two paths differ only in the order
of summation:

* per-diagram::

    Σ_td ∫_polytope dⁿs · cp_td · Π_e factor_e^{(td)}(s, t_ext)

* grouped::

    ∫_polytope dⁿs · [ Σ_td cp_td · Π_e factor_e^{(td)}(s, t_ext) ]

Both integrate over the SAME polytope (identical retardation
constraints — prediagram-shared) and the SAME measure.  So
``total_C(*τ)`` must agree to floating-point round-off.

Precision regime
----------------
The per-diagram path runs each subset through an analytic mode-sum
evaluator (Stage 3a polygon for m=2, Stage 3b causal-poset for m≥3,
or pole-residue closure for m∈{0,1}), achieving machine precision.

The prototype grouped path always falls through to
``_integrate_polytope`` (scipy.nquad on the summed fast_callable),
whose default tolerance is ``epsrel=1.49e-8`` (scipy default).

So with default ``QUAD_OPTS`` the agreement floor is ~1e-8 relative.
With ``QUAD_OPTS`` set to ``epsrel=1e-12 epsabs=1e-14`` the agreement
floor drops to machine precision (~1e-15 rel).

This test asserts agreement at the looser scipy-default level (rtol=
1e-6 with a generous absolute floor) so it passes in the default
config, then re-runs with tightened QUAD_OPTS to assert that the
discrepancy is purely a quadrature-precision effect (and not e.g. a
shot-noise / Wick-mapping bookkeeping bug).

Run with::

    sage -python -m pytest tests/test_grouped_vs_perdiag.py -v
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.phase_j_refactor_fixtures._configs import FIXTURES
from tests.phase_j_refactor_fixtures._runner import evaluate


def _fixture_ids():
    return [fx.name for fx in FIXTURES]


@pytest.mark.parametrize('fx', FIXTURES, ids=_fixture_ids())
def test_grouped_matches_perdiag_at_scipy_default(fx):
    """Grouped path agrees with per-diagram to scipy.nquad's default
    tolerance (~1e-8 relative).

    The grouped path uses scipy.nquad on the summed integrand instead
    of the analytic mode-sum closed forms used by the per-diagram path.
    scipy.nquad's default ``epsrel`` is ``1.49e-8``, so the agreement
    floor is bounded by that.  We assert at rtol=1e-6 with a generous
    atol to leave headroom above the quadrature noise.
    """
    cfg_perdiag = replace(fx, use_grouped_phase_j=False)
    cfg_grouped = replace(fx, use_grouped_phase_j=True)

    r_perdiag = evaluate(cfg_perdiag)
    r_grouped = evaluate(cfg_grouped)

    np.testing.assert_allclose(
        r_grouped['C_values'], r_perdiag['C_values'],
        rtol=1e-6, atol=1e-10,
        err_msg=(
            f'Grouped vs per-diagram disagreement on fixture '
            f'{fx.name!r} exceeds scipy.nquad default tolerance.\n'
            f'  τ_probes = {fx.tau_probes}\n'
            f'  perdiag  = {r_perdiag["C_values"].tolist()}\n'
            f'  grouped  = {r_grouped["C_values"].tolist()}\n'
            f'\nIf the rel-diff is in the ~1e-8 range, this is\n'
            f'expected scipy.nquad noise — see test_grouped_matches_\n'
            f'perdiag_with_tight_quadrature for the machine-precision\n'
            f'version of this test.  If much larger, the grouped path\n'
            f'has diverged from the per-diagram bookkeeping (Wick\n'
            f'contraction, prefactor accumulation, or subset\n'
            f'enumeration).'
        ),
    )


@pytest.mark.parametrize('fx', FIXTURES, ids=_fixture_ids())
def test_grouped_matches_perdiag_with_tight_quadrature(fx):
    """With QUAD_OPTS tightened to near-machine precision, the grouped
    path should match the per-diagram path to floating-point round-off.

    This is the strongest version of the equivalence assertion: if it
    fails, the grouped path has a *real* bookkeeping bug (not a
    quadrature-precision artifact).
    """
    from engine.integration.time_domain import final_integral as _fi
    saved = dict(_fi.QUAD_OPTS)
    _fi.QUAD_OPTS = {
        'limit':  400,
        'epsrel': 1e-12,
        'epsabs': 1e-14,
    }
    try:
        cfg_perdiag = replace(fx, use_grouped_phase_j=False)
        cfg_grouped = replace(fx, use_grouped_phase_j=True)

        r_perdiag = evaluate(cfg_perdiag)
        r_grouped = evaluate(cfg_grouped)

        np.testing.assert_allclose(
            r_grouped['C_values'], r_perdiag['C_values'],
            rtol=1e-10, atol=1e-13,
            err_msg=(
                f'Grouped vs per-diagram disagreement on fixture '
                f'{fx.name!r} exceeds tightened quadrature precision.\n'
                f'  τ_probes = {fx.tau_probes}\n'
                f'  perdiag  = {r_perdiag["C_values"].tolist()}\n'
                f'  grouped  = {r_grouped["C_values"].tolist()}\n'
                f'\nThis indicates a real bookkeeping bug in the\n'
                f'grouped path (Wick contraction, prefactor\n'
                f'accumulation, subset enumeration, or δ-edge\n'
                f'substitution) — not a quadrature-precision artifact.'
            ),
        )
    finally:
        _fi.QUAD_OPTS = saved

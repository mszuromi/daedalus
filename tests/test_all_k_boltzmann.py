"""All-k validation of the temporal pipeline against the exact
Boltzmann stationary density.

For the scalar Langevin equation

    dx/dt = -(mu*x + a*x^2 + b*x^3) + sqrt(2T)*eta,   mu = T = 1,

the stationary density is rho(x) ∝ exp(-(x^2/2 + a*x^3/3 + b*x^4/4)),
whose equal-time cumulants have EXACT perturbative series obtainable
by Gaussian-moment expansion.  Loop order maps onto series order
(each loop adds two cubic powers or one quartic power), so each
(k, ell) pipeline total must reproduce the corresponding series
coefficients exactly:

    kappa_3:  tree = -2a,        1-loop = 42ab - 32a^3
    kappa_4:  tree = -6b,        1-loop = 126b^2 + (a^2-sector)

These tests pin the two bugs found June 2026 in the k>=3 1-loop
combinatorics:

1. ``integrate_diagram``'s external-Wick compensation used a
   role-signature ∏N! heuristic that over-divided whenever leaves
   attached to same-role but non-automorphic vertices (x1/2, x1/3
   deficits at k=4; exact divisor is
   ``|Aut(leaves free)| / |Aut(leaves fixed)|`` — see
   ``external_wick_compensation``).
2. ``diagram_signature`` was an incomplete isomorphism invariant: at
   k=3 it collided 4 of the 11 a^3-sector 1-loop classes into 2
   representatives, silently dropping their integrals (now a true
   canonical form — collisions impossible).

k=2 was exact under both bugs (the heuristic coincides with the index
there, and no k=2 collisions occur), so these higher-k anchors are the
regression net that k=2 tests cannot provide.  The per-class
combinatorics were additionally verified against a brute-force labeled
Wick enumeration (scratch/wick_count_k3_a3.py): all 11 a^3 classes
match with ratio exactly 1.

Runtime: k=3 ~2 min, k=4 ~4 min (enumeration dominates).  Marked
``slow``.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from sage.all import SR  # noqa: E402


def _run_pipeline(action, eq, params, k, max_ell, name):
    from msrjd.core.field_theory import FieldTheory
    from pipeline._propagator import build_propagator, compute_poles_and_residues
    from pipeline._diagrams import enumerate_unique_diagrams
    from msrjd.diagrams.symmetry import classify_coefficient_factors
    from msrjd.diagrams.type_assignment import build_field_index_map
    from msrjd.core.vertices import extract_vertex_types, extract_source_types
    from msrjd.integration.time_domain.final_integral import integrate_diagram
    from pipeline.theory import TemporalTheoryBuilder

    b = (TemporalTheoryBuilder(name).physical_field('x')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('T', default=1.0, domain='positive'))
    for p, v in params.items():
        b = b.parameter(p, default=v)
    model = b.set_action_text(action).equation(lhs=eq, rhs='0').build()

    ft = FieldTheory(model, taylor_order=max(k + 2 * max_ell, 4))
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    num_params = {SR.var('mu'): 1.0, SR.var('T'): 1.0, SR.var('xstar1'): 0.0}
    for p, v in params.items():
        num_params[SR.var(p)] = v
    compute_poles_and_residues(prop, num_params, verbose=False)
    ring = list(ft._ns._ring_var_names)
    resp_idx, phys_idx = build_field_index_map(ring, ft._n_tilde)
    ext = [('dx', 1)] * k
    vtypes = extract_vertex_types(ft)
    stypes = extract_source_types(ft)
    ub, _, _ = enumerate_unique_diagrams(
        ft, model, k=k, max_ell=max_ell, external_fields=ext,
        G_ft=prop['G_ft'], resp_idx=resp_idx, phys_idx=phys_idx,
        vtypes=vtypes, stypes=stypes,
        use_cache=False, parallel=False, verbose=False)
    pdic = {kk: prop[kk] for kk in (
        'K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega', 'D_delta',
        't_var', 'omega', 'nf', 'pole_vals', 'C_mats')}
    tvars = [SR.var('t%d' % i) for i in range(1, k + 1)]
    totals = {}
    for ell in sorted(ub.keys()):
        tt = 0.0
        for td in ub[ell]:
            info = classify_coefficient_factors(
                td, [], {'temporal_type': 'white', 'amplitude_params': []})
            pref = SR(info['scalar_prefactor'])
            if abs(float(pref.subs(num_params))) < 1e-14:
                continue
            res = integrate_diagram(td, pdic, pref, tvars,
                                    num_params=num_params,
                                    external_fields=ext)
            tt += complex(res['contribution'](*([0.0] * k))).real
        totals[ell] = tt
    return totals


@pytest.mark.slow
def test_k3_tree_and_1loop_match_boltzmann_series():
    """kappa_3: tree = -2a, 1-loop = 42ab - 32a^3 (exact series)."""
    a, b = 0.05, 0.05
    totals = _run_pipeline(
        'xt*((Dt+mu)*x + a*x^2 + b*x^3) - T*xt^2',
        '(Dt+mu)*x + a*x^2 + b*x^3',
        {'a': a, 'b': b}, k=3, max_ell=1, name='allk-boltz-k3')
    assert abs(totals[0] - (-2 * a)) < 1e-12, totals
    assert abs(totals[1] - (42 * a * b - 32 * a ** 3)) < 1e-12, totals


@pytest.mark.slow
def test_k4_tree_and_1loop_match_boltzmann_series():
    """kappa_4 for the pure-cubic drift (a=0): tree = -6*eps,
    1-loop = 126*eps^2 (hand-verified Boltzmann series)."""
    eps = 0.05
    totals = _run_pipeline(
        'xt*((Dt+mu)*x + eps*x^3) - T*xt^2',
        '(Dt+mu)*x + eps*x^3',
        {'eps': eps}, k=4, max_ell=1, name='allk-boltz-k4')
    assert abs(totals[0] - (-6 * eps)) < 1e-12, totals
    assert abs(totals[1] - (126 * eps ** 2)) < 1e-12, totals


if __name__ == '__main__':
    test_k3_tree_and_1loop_match_boltzmann_series()
    print('k=3 OK')
    test_k4_tree_and_1loop_match_boltzmann_series()
    print('k=4 OK')

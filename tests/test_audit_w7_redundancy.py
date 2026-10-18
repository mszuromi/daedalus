"""
Regression tests for audit wave W7 (redundancy and retirement).

* RED-03: grouped Phase J is retired; ``use_grouped_phase_j=True`` warns and
  runs the per-diagram integrator (same numbers).
* ACC-67: ``compute_correction_td(kernel_groups=...)`` is retired (it used the
  group's SUMMED prefactor for every diagram of the group).
* ACC-68: the frequency-domain reference (``tests/oracles/symbolic.py``)
  raises on a pole whose half-plane it cannot decide instead of dropping it
  (it returned 0 with status 'ok').
* RED-04: the production-unimported modules are out of the library.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR, DiGraph, matrix, I                    # noqa: E402

from engine.core.vertices import VertexType                    # noqa: E402
from engine.diagrams.type_assignment import TypedDiagram       # noqa: E402


# ── ACC-68: the symbolic reference never drops an undecidable pole ─────────

def _tree_td():
    vt = VertexType(SR(1), [('nt', 1)], [('dn', 1)], (1, 1))
    D = DiGraph([(0, 2), (2, 1)])
    return TypedDiagram(
        (D, D.to_undirected(), [0, 1], [2]), {2: vt},
        {(0, 2): (('nt', 1), ('dn', 1)), (2, 1): (('nt', 1), ('dn', 1))},
        {0: ('nt', 1), 1: ('dn', 1)}, {(0, 2): (0, 0), (2, 1): (0, 0)})


def _diag_prop(a1, a2):
    w = SR.var('omega')
    G = matrix(SR, [[1 / (I * w + a1), 0], [0, 1 / (I * w + a2)]])
    return {'G_ft': G, 'G_ft_explicit': True, 'pole_vals': [I * a1, I * a2],
            'nf': 2}


def test_symbolic_reference_refuses_an_undecidable_half_plane():
    from tests.oracles.symbolic import (
        PoleSearchError, build_integrand_stationary, integrate_to_time_domain)
    b1, b2 = SR.var('w7_beta1'), SR.var('w7_beta2')          # no assumption
    ir = build_integrand_stationary(_tree_td(), _diag_prop(b1, b2), k=2)
    with pytest.raises(PoleSearchError, match='undecidable'):
        integrate_to_time_domain(ir)


def test_symbolic_reference_decided_poles_match_the_closed_form():
    """New case (independent reference): with a > 0 decided (domain) or
    numeric, the tree is the closed form of the chain
    G(t) = e^{-a t} Θ(t) convolved with itself: (t2 - t1) e^{-a (t2-t1)}
    Θ(t2 - t1) up to the diagram's prefactor; the positive-domain and the
    numeric runs agree after substitution."""
    from tests.oracles.symbolic import (
        build_integrand_stationary, integrate_to_time_domain)
    a = SR.var('w7_alpha', domain='positive')
    t1, t2 = SR.var('t_1'), SR.var('t_2')
    sym = integrate_to_time_domain(
        build_integrand_stationary(_tree_td(), _diag_prop(a, a), k=2))
    num = integrate_to_time_domain(
        build_integrand_stationary(_tree_td(), _diag_prop(SR(3) / 2,
                                                           SR(3) / 2), k=2))
    assert sym['status'] == num['status'] == 'ok'
    ratios = []
    for (x1, x2) in ((0.0, 0.7), (-0.4, 1.3), (1.1, 0.2), (0.3, 2.9)):
        vs = complex(sym['time_domain_result'].subs(
            {a: SR(3) / 2, t1: x1, t2: x2}).n())
        vn = complex(num['time_domain_result'].subs({t1: x1, t2: x2}).n())
        assert abs(vs - vn) <= 1e-13 * max(1.0, abs(vn))
        d = x2 - x1
        if d > 0:
            ratios.append(vn / (d * np.exp(-1.5 * d)))
        else:
            assert abs(vn) < 1e-14                          # retarded
    # one constant (the diagram's prefactor) at every point: the closed form
    assert abs(ratios[0]) > 0.1
    assert all(abs(r - ratios[0]) < 1e-12 for r in ratios)


# ── ACC-67: compute_correction_td(kernel_groups=...) is retired ───────────

def test_kernel_groups_input_is_retired_and_direct_input_is_additive():
    from tests.oracles.symbolic import (
        build_integrand_stationary, group_diagrams_by_kernel)
    from engine.integration.time_domain.pipeline import compute_correction_td
    from tests.test_time_domain import (
        _propagator_data_2pop_nondiagonal, _tree_k2_cross_population)
    pd = _propagator_data_2pop_nondiagonal()
    td = _tree_k2_cross_population()
    pf = build_integrand_stationary(td, pd, 2)['scalar_prefactor']
    with pytest.raises(TypeError, match='retired'):
        compute_correction_td(kernel_groups=group_diagrams_by_kernel(
            [td, td], pd, k=2), propagator_data=pd, k=2, origin_leaf_idx=1)

    def val(tds, pfs):
        r = compute_correction_td(typed_diagrams=tds, prefactors=pfs,
                                  propagator_data=pd, k=2, num_params=None,
                                  origin_leaf_idx=1)
        return complex(r['total_C'](0.7, 0.0))
    one = val([td], [pf])
    assert one != 0
    # two copies of one diagram = twice the diagram (the retired input gave
    # four times: each copy weighted by the summed prefactor 2 pf)
    assert abs(val([td, td], [pf, pf]) - 2 * one) <= 1e-14 * abs(one)


# ── RED-04: test-only references are not in the library ───────────────────

@pytest.mark.parametrize('mod', [
    'engine.integration.symbolic', 'engine.integration.numerical',
    'engine.integration.spatial.loop_dyson',
    'engine.integration.spatial.generic_evaluator',
    'engine.integration.spatial.loop_parametric',
    'engine.integration.spatial.spatial_reduce',
    'engine.integration.spatial.temporal_integrate',
    'engine.enumeration.degree_scan', 'engine.enumeration.orderly',
    'engine.core.serialize', 'engine.core.propagator',
    'engine.integration.time_domain.grouped_integral',
    'api._grouped_phase_j'])
def test_retired_modules_are_gone(mod):
    with pytest.raises(ImportError):
        importlib.import_module(mod)


def test_retired_delta_grid_helpers_are_gone():
    import engine.integration.time_domain as TD
    import engine.integration.time_domain.final_integral as FI
    for name in ('eval_delta_contributions_on_tau_grid',
                 'eval_delta_contributions_on_2d_grid'):
        assert not hasattr(TD, name) and not hasattr(FI, name)


# ── LOUD-09: G_t_entry substitutes the recorded time variable ─────────────

def test_G_t_entry_uses_the_recorded_time_variable():
    """With symbolic rates whose names sort before the time variable, the
    old name-order guess substituted a RATE.  New case: a 2x2 triangular
    propagator with rates b, c (closed form of the off-diagonal entry
    (e^{-b t} - e^{-c t}) / (c - b))."""
    from engine.integration.time_domain.propagator_td import (
        build_G_t_matrix, G_t_entry)
    b = SR.var('w7_b', domain='positive')
    c = SR.var('w7_c', domain='positive')
    w, t, s = SR.var('omega'), SR.var('w7_t'), SR.var('w7_s')
    G_ft = matrix(SR, [[1 / (I * w + b), 0],
                       [1 / ((I * w + b) * (I * w + c)), 1 / (I * w + c)]])
    C_b = matrix(SR, [[1, 0], [1 / (c - b), 0]])
    C_c = matrix(SR, [[0, 0], [-1 / (c - b), 1]])
    G = build_G_t_matrix({'pole_vals': [I * b, I * c], 'C_mats': [C_b, C_c],
                          'G_ft': G_ft, 'omega': w}, t)
    from sage.all import exp as sexp
    got = G_t_entry(G, 1, 0, s, include_heaviside=False)
    ref = (sexp(-b * s) - sexp(-c * s)) / (c - b)
    assert (got - ref).simplify_full() == 0
    assert G_t_entry(G, 0, 0, s, include_heaviside=False) == sexp(-b * s)

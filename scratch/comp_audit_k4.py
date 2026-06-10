"""Audit of the external-leaf Wick compensation factor at k=4, 1-loop.

For OU + eps*x^3 (action xt*((Dt+mu)*x + eps*x^3) - T*xt^2) the exact
Boltzmann series gives kappa_4 = -6 eps + 126 eps^2, so the ell=1 total
at equal times must be 126*eps^2.  The pipeline currently returns
76.5*eps^2.

Hypothesis: integrate_diagram's `_compensation` (final_integral.py)
divides the sum over external-leaf mappings by prod N! over leaves
grouped by (vertex_role_signature, field).  The CORRECT divisor is the
number of leaf permutations realizable by graph automorphisms,

    comp_exact = |Aut(Gamma, leaves free)| / |Aut(Gamma, leaves fixed)|

(orbit-stabilizer: summing the integrand over all field-respecting
leaf->position mappings counts each pinned-external diagram class
exactly comp_exact times).  Same role signature does NOT imply the
permutation is realizable, so the heuristic over-divides.

This script computes both factors per diagram and predicts the
corrected total = sum val_i * comp_heur_i / comp_exact_i  ==  126 eps^2.
"""
import sys
sys.path.insert(0, '.')
import collections
from math import factorial
import numpy as np
from sage.all import SR
from msrjd.core.field_theory import FieldTheory
from pipeline._propagator import build_propagator, compute_poles_and_residues
from pipeline._diagrams import enumerate_unique_diagrams
from msrjd.diagrams.symmetry import (
    classify_coefficient_factors, combinatorial_factor,
    vertex_role_signature, _automorphism_order, _wick_leg_factor,
)
from msrjd.diagrams.type_assignment import build_field_index_map
from msrjd.core.vertices import (
    extract_vertex_types, extract_source_types, SourceType, VertexType,
)
from msrjd.integration.time_domain.final_integral import integrate_diagram

mu, T, eps = 1.0, 1.0, 0.05
from pipeline.theory import TemporalTheoryBuilder
model = (TemporalTheoryBuilder('ou-cubic-k4-compaudit')
         .physical_field('x')
         .parameter('mu', default=mu, domain='positive')
         .parameter('eps', default=eps)
         .parameter('T', default=T, domain='positive')
         .set_action_text('xt*((Dt+mu)*x + eps*x^3) - T*xt^2')
         .equation(lhs='(Dt+mu)*x + eps*x^3', rhs='0').build())
ft = FieldTheory(model, taylor_order=4); ft.expand()
prop = build_propagator(ft, model, use_cache=False, verbose=False)
num_params = {SR.var('mu'): mu, SR.var('eps'): eps, SR.var('T'): T,
              SR.var('xstar1'): 0.0}
compute_poles_and_residues(prop, num_params, verbose=False)
ring = list(ft._ns._ring_var_names)
resp_idx, phys_idx = build_field_index_map(ring, ft._n_tilde)
ext = [('dx', 1)] * 4
vtypes = extract_vertex_types(ft)
stypes = extract_source_types(ft)
ub, mb, _ = enumerate_unique_diagrams(ft, model, k=4, max_ell=1,
                                      external_fields=ext, G_ft=prop['G_ft'],
                                      resp_idx=resp_idx, phys_idx=phys_idx,
                                      vtypes=vtypes, stypes=stypes,
                                      use_cache=False, parallel=False,
                                      verbose=False)
pd = {kk: prop[kk] for kk in ('K_ker', 'K_ft', 'G_ft', 'adj_ft', 'D_omega',
                              'D_delta', 't_var', 'omega', 'nf',
                              'pole_vals', 'C_mats')}


def heuristic_comp(td):
    """Replicate final_integral.py's current compensation factor."""
    leaves = list(td.prediagram[2])
    leaf_set = set(leaves)
    vertex_of_leaf = {}
    for ek in td.edge_types:
        u, v = ek[0], ek[1]
        if u in leaf_set and v not in leaf_set:
            vertex_of_leaf[u] = v
        elif v in leaf_set and u not in leaf_set:
            vertex_of_leaf[v] = u
    sig_field_counts = {}
    for lf in leaves:
        v = vertex_of_leaf.get(lf)
        if v is None:
            continue
        sig = vertex_role_signature(v, td)
        field = td.external_legs.get(lf)
        sig_field_counts.setdefault(sig, {}).setdefault(field, 0)
        sig_field_counts[sig][field] += 1
    comp = 1
    for sig, fcounts in sig_field_counts.items():
        for field, count in fcounts.items():
            comp *= factorial(count)
    return comp


def exact_comp(td):
    aut_free = _automorphism_order(td, fix_external=False)
    aut_fixed = _automorphism_order(td, fix_external=True)
    assert aut_free % aut_fixed == 0
    return aut_free // aut_fixed


def edge_summary(td):
    """Compact human-readable wiring summary."""
    leaf_set = set(td.prediagram[2])
    names = {}
    for v, a in td.vertex_assignments.items():
        if isinstance(a, VertexType):
            names[v] = 'c%s' % v
        elif isinstance(a, SourceType):
            names[v] = 'n%s' % v
    for i, lf in enumerate(td.prediagram[2]):
        names[lf] = 'e%d' % i
    parts = []
    for ek in sorted(td.edge_types, key=str):
        u, v = ek[0], ek[1]
        parts.append('%s>%s' % (names.get(u, u), names.get(v, v)))
    return ' '.join(parts)


tvars = [SR.var('t%d' % i) for i in (1, 2, 3, 4)]
tot_now, tot_fix = 0.0, 0.0
rows = []
for td, mult in zip(ub.get(1, []), mb.get(1, [1] * len(ub.get(1, [])))):
    info = classify_coefficient_factors(
        td, [], {'temporal_type': 'white', 'amplitude_params': []})
    pref = SR(info['scalar_prefactor'])
    pn = float(pref.subs(num_params))
    if abs(pn) < 1e-14:
        continue
    res = integrate_diagram(td, pd, pref, tvars,
                            num_params=num_params, external_fields=ext)
    val = complex(res['contribution'](0.0, 0.0, 0.0, 0.0)).real
    ch = heuristic_comp(td)
    ce = exact_comp(td)
    fixed = val * ch / ce
    Scal = combinatorial_factor(td)
    numer = _wick_leg_factor(td)
    af = _automorphism_order(td, fix_external=False)
    ax = _automorphism_order(td, fix_external=True)
    rows.append((val / eps**2, fixed / eps**2, ch, ce, mult, Scal, numer,
                 af, ax, edge_summary(td)))
    tot_now += val
    tot_fix += fixed

print('val/eps^2  fixed/eps^2  comp_heur  comp_exact  mult  Scal  numer  |Aut_free|  |Aut_fix|')
for r in sorted(rows, key=lambda r: -abs(r[1])):
    print('%9.3f  %11.3f  %9d  %10d  %4d  %4d  %5d  %10d  %9d' % r[:9])
    print('    wiring: %s' % r[9])
print()
print('TOTAL current = %.6f  (= %.2f eps^2)' % (tot_now, tot_now / eps**2))
print('TOTAL fixed   = %.6f  (= %.2f eps^2)' % (tot_fix, tot_fix / eps**2))
print('needed        = %.6f  (= 126 eps^2)' % (126 * eps**2))
print('VERDICT:', 'MATCH' if abs(tot_fix - 126 * eps**2) < 1e-9 else
      'MISMATCH (diff %.3e)' % (tot_fix - 126 * eps**2))

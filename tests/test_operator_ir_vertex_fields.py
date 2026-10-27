"""
tests/test_operator_ir_vertex_fields.py
=======================================
Derivative vertices of DIFFERENT fields with the same physical-leg count stay
apart in the spatial operator IR.

The form-factor table (``ns._operator_ir_vertex_terms``) used to match a
diagram node by its leg count alone.  Two decoupled fields ``a`` and ``b``
with ``-c1*Lap(a^2)`` and ``-c2*Lap(b^2)`` (both ``phit*phi^2``) were merged
into one group: each carried the weight ``c_t / (c1 + c2)``, so the one-loop
correlator of ``a`` depended on ``c2``.  (With a multi-species group on one
field the table refused the model.)  Each entry now carries ``fields`` -- the
sorted propagator-matrix columns of the vertex's physical legs -- and a node
only takes the entries of its own field content.

Reference: a decoupled field's cumulants are those of the one-field model.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import SR

from daedalus.api import compute_cumulants
from daedalus.api.model import ModelBuilder

_P = {'ma': 1.0, 'mb': 1.6, 'DD': 1.0, 'c1': 0.3, 'c2': 0.2, 'c3': 0.15, 'T': 1.0}
_A = '- c1*Lap(a^2)'
_B = '- c2*Lap(b^2)'


def _two(a_term=_A, b_term=_B, fields=('a', 'b')):
    """Two decoupled fields, each with its own derivative vertex."""
    b = ModelBuilder('vertex fields', n_populations=0)
    for f in fields:
        b = b.physical_field(f, spatial_dim=1)
    for k, v in _P.items():
        b = b.parameter(k, default=v, domain='positive' if k in ('ma', 'mb', 'DD', 'T') else 'real')
    act = ''
    for f, m, term in zip(fields, ('ma', 'mb'), (a_term, b_term)):
        b = b.mf_residual(lhs=f'(Dt + {m} - DD*Laplacian)*{f}', rhs='0')
        act += (f'{f}t*(Dt({f}) + {m}*{f} - DD*Lap({f}) {term}) - T*{f}t^2 + ')
    return b.set_action_text(act[:-3]).operator_ir().boundary('infinite').initial('stationary').build()


def _table(m):
    from daedalus.engine.core.field_theory import FieldTheory
    ft = FieldTheory(m, taylor_order=4)
    ft.expand()
    sub = {SR.var(k): v for k, v in _P.items()}
    sub[SR.var('astar1')] = sub[SR.var('bstar1')] = 0.0
    return [(t['fields'], t['n_phys'], t['chain'], t['mode'], float(SR(t['weight']).subs(sub)))
            for t in ft._ns._operator_ir_vertex_terms]


def test_same_leg_count_different_fields_are_separate_entries():
    tab = _table(_two())
    assert sorted(tab) == [((0, 0), 2, (('Lap',),), 'composite', 1.0),
                           ((1, 1), 2, (('Lap',),), 'composite', 1.0)]


def test_multi_species_group_next_to_another_field_is_no_longer_refused():
    """c1*Lap(a^2) + c3*a^2 on a (two species), c2*Dx(b)^2 on b: the same leg
    count, different fields (this used to raise NotImplementedError)."""
    tab = _table(_two(a_term='- c1*Lap(a^2) - c3*a^2', b_term='- c2*Dx(b, 0)^2'))
    by = {}
    for fields, n, chain, mode, w in tab:
        by.setdefault(fields, []).append((chain, mode, w))
    assert set(by) == {(0, 0), (1, 1)}
    assert by[(1, 1)] == [((('Dx', 0),), 'perleg', 1.0)]        # a lone species: weight 1
    assert sorted(by[(0, 0)]) == sorted([((('Lap',),), 'composite', pytest.approx(-0.3 / np.hypot(0.3, 0.15))),
                                         ((), 'composite', pytest.approx(-0.15 / np.hypot(0.3, 0.15)))])


def _k2(m, f, ell=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m, k=2, max_ell=ell, external_fields=[(f, 1), (f, 1)],
                              fundamental=_P, tau_grid=np.array([0.0]),
                              spatial_grid=np.array([0.0, 0.5, 1.0]), use_cache=False,
                              parallel=False, verbose=False)
    return {el: np.asarray(v, dtype=float).ravel() for el, v in r['C_tau_x_by_ell'].items()}


def _alone(term, f):
    """The one-field model of field ``f`` (the other field absent)."""
    b = (ModelBuilder('vertex fields alone', n_populations=0).physical_field(f, spatial_dim=1))
    m = 'ma' if f == 'a' else 'mb'
    for k, v in _P.items():
        b = b.parameter(k, default=v, domain='positive' if k in ('ma', 'mb', 'DD', 'T') else 'real')
    return (b.mf_residual(lhs=f'(Dt + {m} - DD*Laplacian)*{f}', rhs='0')
            .set_action_text(f'{f}t*(Dt({f}) + {m}*{f} - DD*Lap({f}) {term}) - T*{f}t^2')
            .operator_ir().boundary('infinite').initial('stationary').build())


_DXA, _DXB = '- c1*Dx(a, 0)^2', '- c2*Dx(b, 0)^2'


@pytest.mark.parametrize('a_term,b_term', [(_A, _B), (_A, _DXB), (_DXA, _B)])
def test_decoupled_field_loop_equals_its_one_field_model(a_term, b_term):
    """The one-loop correlator of each field does not see the other field's
    vertex (before: two different vertex types at one leg count were mixed with
    the weights c_t/(c1 + c2) -- refused as UV-divergent; the same type twice
    only happened to come out right)."""
    m = _two(a_term=a_term, b_term=b_term)
    for f, term in (('a', a_term), ('b', b_term)):
        got, ref = _k2(m, f), _k2(_alone(term, f), f)
        for ell in (0, 1):
            assert np.max(np.abs(ref[ell])) > 0
            np.testing.assert_allclose(got[ell], ref[ell], rtol=1e-12, atol=1e-18)


def test_decoupled_k3_tree_equals_its_one_field_model():
    pts = np.array([[[0.5, 0.3], [-0.4, 0.7]], [[1.2, 0.4], [0.2, 0.9]]])

    def k3(m, f):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = compute_cumulants(m, k=3, max_ell=0, external_fields=[(f, 1)] * 3,
                                  fundamental=_P, spatial_points=pts, use_cache=False,
                                  parallel=False, verbose=False)
        return np.asarray(r['C_kpoint'], dtype=float).ravel()
    m = _two(b_term='- c2*Dx(b, 0)^2')
    for f, term in (('a', _A), ('b', '- c2*Dx(b, 0)^2')):
        ref = k3(_alone(term, f), f)
        assert np.max(np.abs(ref)) > 0
        np.testing.assert_allclose(k3(m, f), ref, rtol=1e-12, atol=1e-18)

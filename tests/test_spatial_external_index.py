"""
tests/test_spatial_external_index.py
====================================
The k=2 mirror completion of the spatial integrator must agree with the
external-assignment rule of the Feynman rules (paper App. A1g,
Prop. prop_ext_compensation): typed diagrams are deduplicated with the leaves
FREE, and each pinned diagram is recovered by summing over the leaf
assignments and dividing by the index ``[Aut : Aut_ext]``.  At k=2 that is
``Γ(τ)+Γ(−τ)`` for index 1 and ``Γ(τ)`` for index 2.

  * every Allen–Cahn φ⁴ record at ℓ ≤ 2: the k=2 shortcut
    (``diagram_correlator_x``) equals the general-k mapping sum
    (``diagram_correlator_pts`` over both assignments ÷ index);
  * the descriptor carries the index, and it is 1 exactly for the records
    whose two orientations are distinct pinned diagrams;
  * ℓ = 3 (slow, ``SPATIAL_SLOW_TESTS=1``): the six ``{R,R}`` classes of
    index 1 — the case the former edge-kind rule got wrong — now carry their
    mirror, and the two paths still agree.

Run:  sage -python -m pytest tests/test_spatial_external_index.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO = os.path.join(os.path.dirname(__file__), '..')

from engine.diagrams.symmetry import external_wick_compensation
from engine.integration.spatial.diagram_descriptor import CEdge, CStackDiagram, diagram_to_cstack
from engine.integration.spatial.full_integrator import (
    _needs_mirror, diagram_correlator_pts, diagram_correlator_x,
    field_respecting_mappings,
)
from engine.integration.spatial.pipeline_bridge import build_pipeline_records, _legs_to_phys_idx
from engine.diagrams.type_assignment import build_field_index_map


def _records(max_ell):
    from sage.all import SR
    from api._propagator import build_propagator
    from api.compute import FieldTheory
    path = os.path.join(_REPO, 'models', 'allen_cahn_1d_subcritical_infinite.model.py')
    spec = importlib.util.spec_from_file_location('ac', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = mod.build()
    ft = FieldTheory(b, taylor_order=2 + 2 * max_ell)
    ft.expand()
    ft.sanity_check(verbose=False)
    prop = build_propagator(ft, b, use_cache=False, verbose=False)
    _, pidx = build_field_index_map(list(ft._ns._ring_var_names), ft._n_tilde)
    ext = _legs_to_phys_idx([('phi', 1), ('phi', 1)], pidx)
    by_ell = build_pipeline_records(ft, b, prop, ext, max_ell=max_ell, verbose=False)
    base = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('T'): 1.0,
            SR.var('lam'): 0.1, SR.var('phistar1'): 0.0}
    live = []
    for el, recs in sorted(by_ell.items()):
        for td, pre in recs:
            pv = float(SR(pre).subs(base))
            if abs(pv) > 1e-14:
                live.append((el, td, pv))
    return live


@pytest.fixture(scope='module')
def live2():
    return _records(2)


_XS = np.array([0.0, 0.4, 1.1])
_TAU = 0.7
_KW = dict(n_t=6, n_s=6)                     # coarse: we test identities, not accuracy


def _both_paths(td, pv, tau=_TAU):
    dd = diagram_to_cstack(td)
    shortcut = diagram_correlator_x(dd, pv, _XS, tau, 1.0, 1.0, spatial_dim=1, **_KW)
    maps = field_respecting_mappings(['phi', 'phi'], ['phi', 'phi'])
    assert len(maps) == 2
    x_pts = np.stack([_XS, np.zeros_like(_XS)], axis=1)     # slot 0 at x, slot 1 at 0
    general = diagram_correlator_pts(dd, pv, x_pts, np.array([0.0, tau]), 1.0, 1.0,
                                     spatial_dim=1, mappings=maps,
                                     comp=external_wick_compensation(td), **_KW)
    return dd, shortcut, general


def test_index_on_descriptor_matches_symmetry(live2):
    for _el, td, _pv in live2:
        dd = diagram_to_cstack(td)
        assert dd.ext_index == external_wick_compensation(td)
        assert dd.ext_index in (1, 2)


def test_k2_shortcut_equals_mapping_sum_ell_le_2(live2):
    """The k=2 completion rule is the general-k assignment rule specialized."""
    assert len(live2) >= 5
    for el, td, pv in live2:
        dd, shortcut, general = _both_paths(td, pv)
        assert np.allclose(shortcut, general, rtol=1e-10, atol=1e-14), \
            f'ell={el} ext_index={dd.ext_index}: {shortcut} vs {general}'


def test_index_two_records_are_tau_even(live2):
    """A leaf-swapping automorphism makes the kinematic τ-even, so counting the
    record once at +τ is exact for index 2."""
    seen = 0
    for _el, td, pv in live2:
        dd = diagram_to_cstack(td)
        if dd.ext_index != 2 or not dd.internal_vertices:
            continue
        seen += 1
        plus = diagram_correlator_x(dd, pv, _XS, _TAU, 1.0, 1.0, **_KW)
        minus = diagram_correlator_x(dd, pv, _XS, -_TAU, 1.0, 1.0, **_KW)
        assert np.allclose(plus, minus, rtol=1e-10)
    assert seen >= 2                              # the two {R,R} ℓ=2 classes


def test_hand_built_descriptor_falls_back_to_kind_rule():
    """No typed diagram ⇒ ``ext_index is None`` ⇒ the former edge-kind rule."""
    cr = CStackDiagram(
        internal_vertices=(2,), external_legs=(0, 1), n_loops=1,
        edges=(CEdge(a=(0.0,), b=(1.0,), kind='C', u=0, v=2, external=True),
               CEdge(a=(0.0,), b=(1.0,), kind='R', u=2, v=1, external=True),
               CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=2, external=False)))
    rr = CStackDiagram(
        internal_vertices=(2,), external_legs=(0, 1), n_loops=1,
        edges=(CEdge(a=(0.0,), b=(1.0,), kind='R', u=0, v=2, external=True),
               CEdge(a=(0.0,), b=(1.0,), kind='R', u=2, v=1, external=True),
               CEdge(a=(1.0,), b=(0.0,), kind='C', u=2, v=2, external=False)))
    assert cr.ext_index is None and _needs_mirror(cr)
    assert rr.ext_index is None and not _needs_mirror(rr)


@pytest.mark.skipif(os.environ.get('SPATIAL_SLOW_TESTS') != '1',
                    reason='ℓ=3 enumeration takes ~2 min; set SPATIAL_SLOW_TESTS=1')
def test_k2_ell3_index_one_RR_records_carry_their_mirror():
    live3 = [r for r in _records(3) if r[0] == 3]
    rr_idx1 = []
    for el, td, pv in live3:
        dd = diagram_to_cstack(td)
        kinds = sorted(e.kind for e in dd.edges if e.external)
        if kinds == ['R', 'R'] and dd.ext_index == 1:
            rr_idx1.append((td, pv))
    assert len(rr_idx1) == 6, len(rr_idx1)        # the classes the kind rule missed
    for td, pv in rr_idx1:
        dd, shortcut, general = _both_paths(td, pv)
        assert np.allclose(shortcut, general, rtol=1e-10, atol=1e-14)
        # and the completion is genuinely two terms, not a kind-rule single Γ(τ)
        single = diagram_correlator_pts(
            dd, pv, np.stack([_XS, np.zeros_like(_XS)], axis=1),
            np.array([0.0, _TAU]), 1.0, 1.0, spatial_dim=1,
            mappings=[(0, 1)], comp=1, **_KW)
        assert not np.allclose(shortcut, single, rtol=1e-6)

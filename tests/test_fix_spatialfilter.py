"""
tests/test_fix_spatialfilter.py
===============================
Regression test for the spatial live-prefactor diagram filter
(``engine.integration.spatial.pipeline_bridge``).

The diagram loop substitutes the supplied params into each diagram's scalar
prefactor and converts to a float::

    try:    pv = float(SR(pre).subs(base_np_sr))
    except (TypeError, ValueError):    continue   # 'q-dependent prefactor'

The ``except`` branch used to SILENTLY drop the diagram whenever the prefactor
did not reduce to a float — including when a model COUPLING was simply absent
from ``base_np_sr``.  That under-counted the correlator with no error (or, when
the dropped diagram was the only loop term, raised the *misleading* "no live
loop diagrams" rather than naming the missing parameter).

The scalar prefactor is q-independent by construction (the loop momentum lives
in the form factors / Symanzik integral, NOT in the scalar prefactor —
instrumentation over every tracked spatial theory at ``max_ell=1`` showed this
branch NEVER fires with complete params).  So a leftover free symbol that names
a model parameter ⇒ that parameter is MISSING ⇒ we now RAISE a clear
``missing parameter <name>`` error instead of dropping the diagram.

A second fix replaces the absolute live-filter threshold
``abs(pv) > 1e-14`` (which dropped a *legitimately tiny* coupling) with an
exact symbolic ``is_zero`` test.

Run:  sage -python -m pytest tests/test_fix_spatialfilter.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
from sage.all import SR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO = os.path.join(os.path.dirname(__file__), '..')

from api.compute import FieldTheory
from api._propagator import build_propagator
from engine.integration.spatial.heat_kernel import SpatialPropagatorError
from engine.integration.spatial.pipeline_bridge import (
    compute_spatial_correlator_generic,
)


def _allen_cahn_model():
    """The shipped 1D stochastic Allen-Cahn (φ⁴, μ>0, infinite domain) theory.

    Its single ``ell=1`` loop diagram (φ³ Hartree from the quartic vertex
    expanded around φ*=0) carries the scalar prefactor ``-12·T²·λ`` — i.e. the
    vertex coupling ``λ`` (``lam``) lives ONLY in the prefactor, not in the
    quadratic propagator, so the propagator builds fine even when ``lam`` is
    omitted.  That is exactly the seam the fix guards.
    """
    path = os.path.join(_REPO, 'theories',
                        'allen_cahn_1d_subcritical_infinite.theory.py')
    spec = importlib.util.spec_from_file_location('ac_thy', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _build():
    model = _allen_cahn_model()
    ft = FieldTheory(model, taylor_order=4)
    ft.expand()
    prop = build_propagator(ft, model, use_cache=False, verbose=False)
    ext = [('phi', 1), ('phi', 1)]
    tau = np.array([0.0])
    xg = np.linspace(0.0, 4.0, 5)
    return model, ft, prop, ext, tau, xg


def test_missing_vertex_coupling_raises_not_silent_undercount():
    """An INCOMPLETE params dict (the φ³ vertex coupling ``lam`` omitted) must
    RAISE a clear 'missing parameter lam' error — NOT silently drop the
    ``-12·T²·λ`` diagram (which would under-count the correlator)."""
    model, ft, prop, ext, tau, xg = _build()
    # Propagator-relevant params present; the VERTEX coupling `lam` omitted.
    incomplete = {SR.var('mu'): 1.0, SR.var('D'): 1.0,
                  SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    with pytest.raises(SpatialPropagatorError) as exc:
        compute_spatial_correlator_generic(
            ft, model, prop, incomplete, ext, tau, xg,
            max_ell=1, parallel=False)
    msg = str(exc.value)
    # The error must NAME the missing parameter, not give the misleading
    # "no live loop diagrams" (which is what the un-fixed code produced).
    assert 'missing parameter' in msg, msg
    assert 'lam' in msg, msg


def test_complete_params_still_run():
    """With COMPLETE params the loop correlator runs (no spurious raise) and
    keeps the live diagram — preserving the prior passing behavior."""
    model, ft, prop, ext, tau, xg = _build()
    full = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('lam'): 0.1,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    C, info = compute_spatial_correlator_generic(
        ft, model, prop, full, ext, tau, xg, max_ell=1, parallel=False)
    assert info.get('n_live_diagrams', 0) >= 1
    assert np.asarray(C).shape == (len(tau), len(xg))


def test_tiny_coupling_is_not_dropped_by_absolute_threshold():
    """A legitimately TINY coupling (prefactor ``-12·T²·λ`` ≈ 1.2e-15 for
    λ=1e-16, BELOW the old 1e-14 absolute cutoff) must be KEPT by the new
    symbolic ``is_zero`` live filter, not dropped as numerically-zero."""
    model, ft, prop, ext, tau, xg = _build()
    tiny = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('lam'): 1e-16,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    # sanity: the prefactor magnitude is below the retired absolute threshold
    assert abs(-12.0 * 1.0 ** 2 * 1e-16) < 1e-14
    C, info = compute_spatial_correlator_generic(
        ft, model, prop, tiny, ext, tau, xg, max_ell=1, parallel=False)
    assert info.get('n_live_diagrams', 0) >= 1, \
        'tiny but nonzero coupling was dropped by the live filter'


def test_genuinely_zero_coupling_is_still_dropped():
    """An EXACTLY zero coupling (λ=0 ⇒ prefactor symbolically 0) must still be
    dropped — there is then no live loop diagram (the φ⁴ Hartree vanishes)."""
    model, ft, prop, ext, tau, xg = _build()
    zero = {SR.var('mu'): 1.0, SR.var('D'): 1.0, SR.var('lam'): 0.0,
            SR.var('T'): 1.0, SR.var('phistar1'): 0.0}
    with pytest.raises(SpatialPropagatorError) as exc:
        compute_spatial_correlator_generic(
            ft, model, prop, zero, ext, tau, xg, max_ell=1, parallel=False)
    assert 'no live loop diagrams' in str(exc.value)

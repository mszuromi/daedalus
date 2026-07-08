"""Regression: spatial k>=3 over a (chi, tau) GRID.

``dd.run`` with ``chi_grid`` (+ optional ``tau_grid``) and no ``spatial_points``
sweeps the full 2(k-1)-D grid of the k-1 non-anchor legs, reshapes the flat
per-event k-point cumulant into ``result['C_kpoint_grid']`` (axis order
(tau2, chi2, tau3, chi3, ...)), and must agree with explicit ``spatial_points``
at the same geometry.  Convenience over the same evaluator — no new physics.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

import daedalus as dd  # noqa: E402

_P = {'mu': 1.0, 'D': 1.0, 'lam': 0.3, 'T': 1.0}


def test_k3_grid_shape_symmetry_and_matches_explicit():
    model, mod = dd.load_model('kpz_1d')
    # equal-time grid: chi = [0, 0.5, 1, 1.5, 2], single tau=0
    res = dd.run(model, dd.Config(
        k=3, max_ell=0, external_fields=[('dh', 1)] * 3,
        parameters=_P, chi_grid=(0.0, 2.0, 5), tau_grid=[0.0]), mod)
    g = np.real(res['C_kpoint_grid'])
    assert g.shape == (1, 5, 1, 5)            # (tau2, chi2, tau3, chi3)
    assert np.all(np.isfinite(g))
    assert res['kpoint_grid_axes']['n_legs'] == 2
    # kappa_3 is symmetric under exchanging the two identical legs
    assert np.allclose(g[0, :, 0, :], g[0, :, 0, :].T, atol=1e-6)
    # the grid point (chi2=0.5, chi3=1.0) must equal the explicit-events value
    pts = np.array([[[0.5, 0.0], [1.0, 0.0]]])
    ref = dd.run(model, dd.Config(
        k=3, max_ell=0, external_fields=[('dh', 1)] * 3,
        parameters=_P, spatial_points=pts), mod)
    assert np.isclose(np.real(ref['C_kpoint'])[0], g[0, 1, 0, 2], atol=1e-6)


def test_k3_full_chi_tau_grid_is_4d():
    model, mod = dd.load_model('kpz_1d')
    res = dd.run(model, dd.Config(
        k=3, max_ell=0, external_fields=[('dh', 1)] * 3,
        parameters=_P, chi_grid=(0.0, 1.0, 3), tau_grid=(-1.0, 1.0, 3)), mod)
    g = np.asarray(res['C_kpoint_grid'])
    assert g.shape == (3, 3, 3, 3)           # 2(k-1)=4 axes
    assert np.all(np.isfinite(g))


def test_k3_grid_requires_points_or_chi_grid():
    model, mod = dd.load_model('kpz_1d')
    with pytest.raises(ValueError):
        dd.run(model, dd.Config(
            k=3, max_ell=0, external_fields=[('dh', 1)] * 3, parameters=_P), mod)

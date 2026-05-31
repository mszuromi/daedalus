"""
tests/test_spatial_sim_2d.py
============================
Correctness pins for the 2D Langevin simulator (``models.spatial_field_2d_sim``)
— the d=2 backend-C validation oracle.  Validates the linear-theory structure
factor S(k) against the exact lattice spectrum T/ω_k (per-mode and radial), and
the equal-time variance against the lattice sum.

Run:  sage -python -m pytest tests/test_spatial_sim_2d.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.spatial_field_2d_sim import (
    simulate_2d, structure_factor_2d, radial_structure_factor_2d,
    radial_correlator_2d, _dispersion_2d,
)


@pytest.fixture(scope='module')
def _lin2d():
    return simulate_2d(L=16.0, N=40, mu=1.0, D=1.0, T=1.0,
                       n_steps=40000, burn_in=8000, record_every=10, seed=5)


def test_2d_meta_k_max(_lin2d):
    _snaps, meta = _lin2d
    assert abs(meta['k_max'] - math.pi / meta['dx']) <= 1e-12
    assert meta['spatial_dim'] == 2


def test_2d_structure_factor_matches_lattice(_lin2d):
    """Per-mode S(k) = T/ω_k (exact lattice dispersion) in a low/mid-k band —
    confirms the d=2 noise normalization ⟨|φ̂_k|²⟩ = T N⁴/(L² ω_k)."""
    snaps, meta = _lin2d
    kx, ky, S = structure_factor_2d(snaps, meta)
    disp = _dispersion_2d(meta['N'], meta['dx'], meta['mu'], meta['D'])
    pred = meta['T'] / disp
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    band = (kmag > 0.5) & (kmag < 0.5 * meta['k_max'])
    rel = np.abs(S[band] - pred[band]) / pred[band]
    assert np.median(rel) < 0.06           # MC noise; median robust


def test_2d_radial_structure_factor_continuum(_lin2d):
    """Radially-binned S(|k|) → T/(μ+D|k|²) in the low-k continuum band."""
    snaps, meta = _lin2d
    kc, Sr = radial_structure_factor_2d(snaps, meta, n_bins=30)
    cont = meta['T'] / (meta['mu'] + meta['D'] * kc ** 2)
    band = (kc > 0.4) & (kc < 2.0)
    rel = np.abs(Sr[band] - cont[band]) / cont[band]
    assert np.nanmedian(rel) < 0.10


def test_2d_variance_matches_lattice_sum(_lin2d):
    """Equal-time ⟨φ²⟩ vs the exact lattice momentum sum (T/L²)Σ_k 1/ω_k."""
    snaps, meta = _lin2d
    var_sim = float(np.mean(snaps ** 2))
    disp = _dispersion_2d(meta['N'], meta['dx'], meta['mu'], meta['D'])
    var_lat = (meta['T'] / meta['L'] ** 2) * float(np.sum(1.0 / disp))
    assert abs(var_sim - var_lat) <= 0.05 * var_lat


def test_2d_radial_correlator_vs_K0(_lin2d):
    """Real-space radial C(r) vs the exact continuum K₀ correlator
    (T/2πD)K₀(r√(μ/D)) in a band — the direct real-space oracle that
    compute_cumulants' d=2 C(r,0) is compared against."""
    import math
    from scipy.special import k0
    snaps, meta = _lin2d
    rc, Cr = radial_correlator_2d(snaps, meta, n_bins=40)
    kappa = math.sqrt(meta['mu'] / meta['D'])
    Cr_th = (meta['T'] / (2 * math.pi * meta['D'])) * np.array(
        [float(k0(kappa * r)) if r > 0 else np.nan for r in rc])
    band = (rc > 1.0) & (rc < 4.0)
    rel = np.abs(Cr[band] - Cr_th[band]) / Cr_th[band]
    assert np.nanmedian(rel) < 0.12

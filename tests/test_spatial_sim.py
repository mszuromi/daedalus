"""
tests/test_spatial_sim.py
=========================
Smoke + correctness pins for the 1D Langevin simulator
(``models.spatial_field_1d_sim``) — the backend-C validation oracle.  Closes the
"no test imports the simulator" gap from the pre-C0 audit, and validates the W8
additions (``meta['k_max']`` + ``structure_factor``).

Run:  sage -python -m pytest tests/test_spatial_sim.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulations.spatial_field_1d_sim import (
    simulate, structure_factor, lattice_sum_variance,
)


def _lattice_disp(q, mu, D, dx):
    return mu + (2.0 * D / dx ** 2) * (1.0 - np.cos(q * dx))


@pytest.fixture(scope='module')
def _linear_sim():
    # linear (λ=g=0) Gaussian model: S(q) and ⟨φ²⟩ have exact lattice values.
    return simulate(L=20.0, N=128, mu=1.0, D=1.0, T=1.0, lam=0.0,
                    n_steps=120000, burn_in=20000, record_every=10, seed=7)


def test_meta_k_max(_linear_sim):
    """k_max = π/dx = πN/L is reported (the physical UV cutoff)."""
    _snaps, _x, meta = _linear_sim
    assert abs(meta['k_max'] - math.pi / meta['dx']) <= 1e-12
    assert abs(meta['k_max'] - math.pi * meta['N'] / meta['L']) <= 1e-9


def test_structure_factor_matches_lattice_linear(_linear_sim):
    """The linear-model equal-time S(q) equals the exact lattice T/disp(q) in a
    low/mid-q band (the matched-cutoff oracle), within Monte-Carlo noise."""
    snaps, _x, meta = _linear_sim
    q, S = structure_factor(snaps, meta)
    pred = meta['T'] / _lattice_disp(q, meta['mu'], meta['D'], meta['dx'])
    band = (q > 0.2) & (q < 0.6 * meta['k_max'])
    rel = np.abs(S[band] - pred[band]) / pred[band]
    assert np.median(rel) < 0.12          # MC noise; median is robust


def test_variance_matches_lattice_sum(_linear_sim):
    """Real-space ⟨φ²⟩ = mean S/L-style sum matches the exact lattice variance."""
    snaps, _x, meta = _linear_sim
    var_sim = float(np.mean(snaps ** 2))
    var_exact = lattice_sum_variance(meta['L'], meta['N'], meta['mu'],
                                     meta['D'], meta['T'])
    assert abs(var_sim - var_exact) <= 0.05 * var_exact

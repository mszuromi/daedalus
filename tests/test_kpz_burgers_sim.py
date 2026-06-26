"""
tests/test_kpz_burgers_sim.py
=============================
Fast, deterministic (fixed-seed) smoke + physics checks for the KPZ /
Burgers gradient-vertex forcings added to the 1-D spectral simulator
(``simulations/spatial_field_1d_sim.py``).  The quantitative multi-seed sim-vs-
framework validation lives in ``docs/kpz_burgers_sim_validation.py``.

Run::

    sage -python -m pytest tests/test_kpz_burgers_sim.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO = os.path.join(os.path.dirname(__file__), '..')


def _load_sim():
    p = os.path.join(_REPO, 'simulations', 'spatial_field_1d_sim.py')
    spec = importlib.util.spec_from_file_location('sim', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_P = dict(L=16.0, N=64, mu=1.0, D=1.0, T=1.0,
          n_steps=60000, burn_in=10000, record_every=20)


def test_free_variance_matches_lattice_sum():
    sim = _load_sim()
    snaps, _, _ = sim.simulate(seed=5, **_P)
    var = float(np.mean(snaps**2))
    exact = float(sim.lattice_sum_variance(_P['L'], _P['N'], 1.0, 1.0, 1.0))
    assert abs(var - exact) / exact < 0.05, f'free var {var:.4f} vs {exact:.4f}'


def test_kpz_excess_velocity_sign_and_scale():
    """KPZ (∂_xφ)² drives a POSITIVE excess velocity ⟨φ⟩=(λ/2μ)⟨(∂_xφ)²⟩,
    the per-leg q² form factor loop-averaged.  Check sign + ballpark vs the
    lattice tree prediction (a short fixed-seed run ⇒ loose tolerance)."""
    sim = _load_sim()
    lam = 0.5
    snaps, _, _ = sim.simulate(seed=5, lam_kpz=lam, **_P)
    exc_sim = float(np.mean(snaps))
    # Tree-level lattice prediction.
    L, N, dx = _P['L'], _P['N'], _P['L'] / _P['N']
    ks = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    disp = 1.0 + 2.0 * (1.0 - np.cos(ks * dx)) / dx**2
    ddphi2 = (1.0 / L) * np.sum((np.sin(ks * dx) / dx) ** 2 / disp)
    exc_th = (lam / 2.0) * ddphi2
    assert exc_sim > 0.0, 'KPZ excess velocity must be positive (roughening)'
    assert abs(exc_sim - exc_th) / exc_th < 0.20, \
        f'KPZ excess velocity sim {exc_sim:.3f} vs lattice {exc_th:.3f}'


def test_burgers_has_no_excess_velocity():
    """Burgers −(λ/2)∂_x(φ²) is conservative: its k=0 forcing component
    vanishes (ik|_{k=0}=0), so there is NO excess velocity (⟨φ⟩≈0)."""
    sim = _load_sim()
    snaps, _, _ = sim.simulate(seed=5, lam_burg=0.5, **_P)
    # Free-field k=0 mode still fluctuates; the MEAN must stay ~0 (no drift).
    assert abs(float(np.mean(snaps))) < 0.05, \
        f'Burgers spurious excess velocity ⟨φ⟩={float(np.mean(snaps)):.3f}'


def test_vectorized_noise_is_finite_and_stationary():
    """The (vectorized) OU noise + gradient forcings stay finite and the
    field is stationary (no blow-up) for both KPZ and Burgers."""
    sim = _load_sim()
    for kw in ({'lam_kpz': 0.5}, {'lam_burg': 0.5}):
        snaps, _, _ = sim.simulate(seed=9, **_P, **kw)
        assert np.all(np.isfinite(snaps))
        # second half variance ≈ first half (stationary, no drift/blow-up)
        h = snaps.shape[0] // 2
        v1 = float(np.var(snaps[:h])); v2 = float(np.var(snaps[h:]))
        assert abs(v1 - v2) / max(v1, v2) < 0.25


def _load_sim2d():
    p = os.path.join(_REPO, 'simulations', 'spatial_field_2d_sim.py')
    spec = importlib.util.spec_from_file_location('s2', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kpz_d2_excess_velocity():
    """d=2 KPZ (∇h)²=(∂ₓh)²+(∂_yh)²: the excess velocity ⟨φ⟩=(κ/2μ)⟨(∇φ)²⟩ (both
    axes) matches the lattice tree prediction — the clean check of the per-axis
    d=2 form-factor machinery."""
    sim = _load_sim2d()
    L, N, mu, D, T, kpz = 12.0, 32, 1.0, 1.0, 1.0, 0.5
    sn, meta = sim.simulate_2d(L=L, N=N, mu=mu, D=D, T=T, lam_kpz=kpz, dt=0.02,
                               n_steps=40000, burn_in=8000, record_every=20, seed=3)
    exc_sim = float(np.mean(sn))
    dx = L / N
    kd = 2.0 * np.pi * np.fft.fftfreq(N)
    cx, cy = np.cos(kd)[:, None], np.cos(kd)[None, :]
    disp = mu + (2.0 * D / dx**2) * (2.0 - cx - cy)
    sx, sy = (np.sin(kd)[:, None] / dx) ** 2, (np.sin(kd)[None, :] / dx) ** 2
    exc_th = (kpz / (2.0 * mu)) * (T / L**2) * np.sum((sx + sy) / disp)
    assert exc_sim > 0.0, 'KPZ d=2 excess velocity must be positive'
    assert abs(exc_sim - exc_th) / exc_th < 0.15, \
        f'KPZ d=2 excess velocity sim {exc_sim:.3f} vs lattice {exc_th:.3f}'


def test_modelb_kpz_2d_forcings_finite():
    """The d=2 Model B ∇²(φ²) (g_lap) and KPZ (lam_kpz) forcings stay finite and
    stationary (Model B's conserved ∇² is stiff → smaller dt)."""
    sim = _load_sim2d()
    P = dict(L=12.0, N=32, mu=1.0, D=1.0, T=1.0, n_steps=20000, burn_in=5000,
             record_every=20, seed=7)
    for kw in ({'g_lap': 0.1, 'dt': 0.01}, {'lam_kpz': 0.4, 'dt': 0.02}):
        sn, _ = sim.simulate_2d(**P, **kw)
        assert np.all(np.isfinite(sn))
        h = sn.shape[0] // 2
        v1, v2 = float(np.var(sn[:h])), float(np.var(sn[h:]))
        assert abs(v1 - v2) / max(v1, v2) < 0.3

"""
Spatial v1 — simulator vs framework cross-check  [2026-05-29]
============================================================
Independent physical validation of the tree-level spatial correlator
``compute_cumulants(spatial_grid=…)`` against a direct Langevin
simulation (``models/spatial_field_1d_sim.py``, spectral ETD1).

This complements the machine-precision closed-form validation in
``tests/test_spatial_correlator.py`` with a Monte-Carlo cross-check —
the gold-standard physical sanity test.

RESULT (sage -python docs/spatial_spikes/spatial_sim_vs_framework_validation.py):

  (1) LINEAR (λ=0), L=20, N=200, dx=0.1, 20k samples:
        sim   ⟨φ²⟩ = 0.49589
        lattice sum  = 0.49938   (0.70%  — sampling noise)
        continuum    = 0.50000   (0.82%)
      The spectral exponential integrator is unbiased in dt, so the
      residual is sampling noise (≈1/√N_samp), NOT a systematic gap.
      (Plain Euler-Maruyama gave a 1.9% systematic excess from the
      fast-mode dt bias — fixed by the ETD scheme.)

  (3) framework C(x,0) vs simulator (λ=0, PBC L=20):
        x=0: 0.82% ; x=1: 1.4% ; x=2: 2.7% ; x≥3: 5-9%
      (relative error grows where C(x) is tiny — ~0.003 at x=5 — and
      sampling-noise-dominated; absolute agreement stays good.)

  (4) Allen-Cahn λ=0.3 (full nonlinear sim vs tree-level/Gaussian):
        sim ⟨φ²⟩  = 0.42896
        tree ⟨φ²⟩ = 0.50000
        Δ (1-loop+) = -0.071   ← negative, as expected: the λφ³
      nonlinearity suppresses fluctuations.  This is exactly the
      correction the (deferred) Phase-5b loop integration would
      compute; the tree-level framework result is its leading term.

VERDICT: the tree-level spatial correlator agrees with direct
simulation within sampling noise for the linear theory, and the
Allen-Cahn nonlinear correction has the right sign + sensible
magnitude.  Combined with the closed-form checks, the spatial
propagator + tree-level correlator are validated.

Reproduce: ~1-2 min (two 200k-step spectral simulations + framework).
"""
import sys
import math
import os

_REPO = '/Users/matthewszuromi/Documents/Education/BU PhD/Ocker Lab/Automated Feynman Calculations'
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
from models.spatial_field_1d_sim import (
    simulate, equal_time_correlator, lattice_sum_variance)

L, N, mu, D, T = 20.0, 200, 1.0, 1.0, 1.0

print('=== (1) LINEAR (lam=0) self-consistency + continuum ===')
snaps, xg, meta = simulate(L=L, N=N, mu=mu, D=D, lam=0.0, T=T,
                           n_steps=200000, burn_in=20000,
                           record_every=10, seed=7)
print('  meta:', {k: (round(v, 5) if isinstance(v, float) else v)
                  for k, v in meta.items()})
var_sim = float(np.mean(snaps**2))
var_lat = lattice_sum_variance(L, N, mu, D, T)
var_coth = T / (2 * math.sqrt(mu * D)) * (1.0 / math.tanh((L / 2) * math.sqrt(mu / D)))
print(f'  sim   <phi^2> = {var_sim:.5f}')
print(f'  lattice sum   = {var_lat:.5f}  (rel {abs(var_sim-var_lat)/var_lat:.2%})')
print(f'  continuum coth= {var_coth:.5f}  (rel {abs(var_sim-var_coth)/var_coth:.2%})')

C_sim = equal_time_correlator(snaps)
dx = meta['dx']
from pipeline import compute_cumulants
import importlib.util


def _load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.build()


modelp = _load('theories/allen_cahn_1d_subcritical_pbc.theory.py')
xs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
th = compute_cumulants(model=modelp, k=2, max_ell=0,
                       fundamental={'mu': mu, 'D': D, 'lam': 0.0, 'T': T, 'L': L},
                       external_fields=[('phi', 1), ('phi', 2)],
                       tau_max=1.0, tau_step=1.0, spatial_grid=xs,
                       parallel=False, verbose=False, use_cache=True)
it0 = int(np.argmin(np.abs(th['tau_grid'])))
print('\n=== (3) framework C(x,0) vs simulator (lam=0, PBC L=20) ===')
print(f'  {"x":>4} {"sim":>10} {"framework":>10} {"rel":>8}')
for x in xs:
    m = int(round(x / dx))
    c_fw = th['C_tau_x'][it0, list(xs).index(x)].real
    print(f'  {x:>4.0f} {C_sim[m]:>10.5f} {c_fw:>10.5f} '
          f'{abs(C_sim[m]-c_fw)/max(abs(c_fw),1e-9):>7.2%}')

print('\n=== (4) Allen-Cahn lam=0.3: sim (nonlinear) vs tree (Gaussian) ===')
snaps2, _, _ = simulate(L=L, N=N, mu=mu, D=D, lam=0.3, T=T,
                        n_steps=200000, burn_in=20000, record_every=10, seed=11)
var_sim2 = float(np.mean(snaps2**2))
th2 = compute_cumulants(model=modelp, k=2, max_ell=0,
                        fundamental={'mu': mu, 'D': D, 'lam': 0.3, 'T': T, 'L': L},
                        external_fields=[('phi', 1), ('phi', 2)],
                        tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0]),
                        parallel=False, verbose=False, use_cache=True)
var_tree = th2['C_tau_x'][int(np.argmin(np.abs(th2['tau_grid']))), 0].real
print(f'  sim <phi^2> (lam=0.3) = {var_sim2:.5f}')
print(f'  tree <phi^2>          = {var_tree:.5f}')
print(f'  1-loop+ correction    = {var_sim2-var_tree:+.5f}  (expect <0)')

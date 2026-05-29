"""
Stage C — TEST the spatial 1-loop against a direct simulation.

Compares, for the stochastic Allen-Cahn theory (∂_t φ = -(μ-D∇²)φ - λφ³ + ξ,
⟨ξξ⟩=2T):

  * framework TREE   ⟨φ²⟩₀   = compute_cumulants(max_ell=0)
  * framework 1-LOOP ⟨φ²⟩    = compute_cumulants(max_ell=1)   [strict O(λ)]
  * framework HARTREE ⟨φ²⟩   = T/(2√(A_eff D))  (resummed, from sp_info A_eff)
  * direct SIMULATION (full nonlinear λφ³)  models/spatial_field_1d_sim

μ=D=T=1.  Tree ⟨φ²⟩₀ = 0.5.  Expect: 1-loop < tree (suppression); Hartree
tracks the sim; strict-1-loop is the leading-O(λ) term (further from sim as
λ grows, as it should be).
"""
import importlib.util, os, sys
sys.path.insert(0, '.')
import numpy as np
from pipeline import compute_cumulants
from models.spatial_field_1d_sim import simulate

mu = D = T = 1.0


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


model = load('theories/allen_cahn_1d_subcritical_infinite.theory.py')


def framework(lam, max_ell):
    th = compute_cumulants(
        model=model, k=2, max_ell=max_ell,
        fundamental={'mu': mu, 'D': D, 'lam': lam, 'T': T},
        external_fields=[('phi', 1), ('phi', 2)], tau_max=1.0, tau_step=1.0,
        spatial_grid=np.array([0.0]), parallel=False, verbose=False,
        use_cache=False)
    i0 = int(np.argmin(np.abs(th['tau_grid'])))
    phi2 = float(np.asarray(th['C_tau_x']).real[i0, 0])
    return phi2, th.get('spatial_info', {})


print(f'{"λ":>5} {"tree":>8} {"1-loop":>9} {"Hartree":>9} {"SIM":>9} '
      f'{"g=M·λ":>7} {"1loop-sim":>10}')
for lam in [0.2, 0.4]:
    tree, _ = framework(lam, 0)
    loop, si = framework(lam, 1)
    A_eff = si.get('A_eff_hartree')
    hartree = T / (2 * np.sqrt(A_eff * D)) if A_eff else float('nan')
    snaps, _, _ = simulate(L=20.0, N=200, mu=mu, D=D, lam=lam, T=T,
                           dt=0.02, n_steps=150000, burn_in=15000,
                           record_every=10, seed=7)
    sim = float(np.mean(snaps ** 2))
    print(f'{lam:>5} {tree:>8.4f} {loop:>9.4f} {hartree:>9.4f} {sim:>9.4f} '
          f'{si.get("self_energy_coeff_g", float("nan")):>7.4f} '
          f'{loop - sim:>10.4f}')

print('\n(g must = 3λ; Hartree ≈ SIM; strict 1-loop is the leading-O(λ) term)')

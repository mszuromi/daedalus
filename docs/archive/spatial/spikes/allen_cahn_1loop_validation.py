"""
Allen-Cahn 1-loop validation target (Stage C reference).

The framework's TREE result ignores λφ³.  The 1-loop tadpole self-energy
is a constant mass shift Σ = 3λ⟨φ²⟩₀ → effective mass μ_eff = μ + Σ.
This script documents the loop TARGET the pipeline integrator must
eventually reproduce, by comparing:

  * framework TREE  ⟨φ²⟩₀ = C(0,0)        (via the real pipeline)
  * strict 1-loop   ⟨φ²⟩ ≈ ⟨φ²⟩₀ + Σ·∂C₀/∂μ  (leading O(λ))
  * self-consistent Hartree  μ_eff = μ + 3λ·T/(2√(μ_eff D))
  * direct SIMULATION (full nonlinear λφ³)  models/spatial_field_1d_sim

μ=D=T=1.  ξ=1.  Tree ⟨φ²⟩₀ = 0.5.
"""
import sys, math, importlib.util
sys.path.insert(0, '.')
import numpy as np
import scipy.optimize as opt
from models.spatial_field_1d_sim import simulate

mu = D = T = 1.0


def load(p):
    s = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.build()


# Framework tree ⟨φ²⟩₀ via the real pipeline (max_ell=0)
from pipeline import compute_cumulants
model = load('theories/allen_cahn_1d_subcritical_infinite.theory.py')
th = compute_cumulants(model=model, k=2, max_ell=0,
                       fundamental={'mu': mu, 'D': D, 'lam': 0.1, 'T': T},
                       external_fields=[('phi', 1), ('phi', 2)],
                       tau_max=1.0, tau_step=1.0, spatial_grid=np.array([0.0]),
                       parallel=False, verbose=False, use_cache=True)
phi2_tree = float(np.asarray(th['C_tau_x']).real[int(np.argmin(np.abs(th['tau_grid']))), 0])
dC0_dmu = -T / (4 * mu**1.5 * math.sqrt(D))   # ∂/∂μ [T/(2√(μD))]
print(f'framework tree ⟨φ²⟩₀ = {phi2_tree:.6f}  (closed 0.5)')
print(f'{"λ":>5} {"sim ⟨φ²⟩":>10} {"strict-1loop":>13} {"Hartree":>9} '
      f'{"sim-tree":>9}')
for lam in [0.2, 0.4]:
    Sigma = 3 * lam * phi2_tree
    strict = phi2_tree + Sigma * dC0_dmu
    f = lambda me: me - (mu + 3 * lam * T / (2 * math.sqrt(me * D)))
    me = opt.brentq(f, 0.3, 6.0)
    hartree = T / (2 * math.sqrt(me * D))
    snaps, _, meta = simulate(L=20.0, N=200, mu=mu, D=D, lam=lam, T=T,
                              dt=0.02, n_steps=200000, burn_in=20000,
                              record_every=10, seed=7)
    sim = float(np.mean(snaps**2))
    print(f'{lam:>5} {sim:>10.5f} {strict:>13.5f} {hartree:>9.5f} '
          f'{sim-phi2_tree:>9.5f}')
print('\n(expect: sim ≈ Hartree; strict-1-loop is the leading O(λ) term;')
print(' sim-tree < 0 = the loop suppression the pipeline integrator must give)')

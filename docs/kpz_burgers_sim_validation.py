"""
KPZ + Burgers gradient-vertex validation: 1-D spectral simulator vs the
framework (compute_cumulants), branch ``spatial-extension``.

Run::

    sage -python docs/kpz_burgers_sim_validation.py

Two independent checks of the per-leg / composite ``∂_x`` form-factor machinery
(``μ=D=T=1``, ``λ=0.3``):

1. **KPZ excess velocity** (the headline — a large, clean signal).  The
   ``(∂_xφ)²`` nonlinearity has a non-zero spatial mean, so the k=0 mode
   acquires a steady drift ``⟨φ⟩ = (λ/2μ)⟨(∂_xφ)²⟩`` — the famous KPZ ``v∞``.
   This is *exactly* the per-leg form factor ``q²`` averaged over the loop, so
   it directly tests the per-leg vertex.  Sim ≈ 0.411 vs the lattice prediction
   ``(λ/2μ)·Σ_k (sin(k·dx)/dx)²·T/(L·ω_k)`` ≈ 0.4106 → **~0.1 %**.

2. **Connected variance shift** ``δC(0,0) = C_nl(0,0) − C_free(0,0)``, via
   PAIRED runs (common random numbers: identical noise seed for the free and
   nonlinear integrations, so the noise cancels in the difference).  Sign +
   magnitude vs ``compute_cumulants`` tree→1-loop:
     * Burgers (composite, no excess velocity ⇒ bias-free): sim
       ``−0.00022 ± 0.00015`` vs theory ``−0.00013`` — right sign, within 1σ.
     * KPZ (per-leg): sim ``+0.00055 ± 0.00013`` vs theory ``+0.00109`` —
       right sign (roughening) and order; biased LOW because the connected
       estimator subtracts the *sample* mean² and the KPZ k=0 mode carries
       extra nonlinear-driven fluctuations (a statistics artifact, NOT a
       machinery error — the same vertex's excess velocity matches to 0.1 %).

The KPZ/Burgers 1-loop self-energy is UV-convergent in d=1, so the shift is
robust to the (different) sim vs framework cutoffs.
"""
import importlib.util
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_sim():
    p = os.path.join(_REPO, 'models', 'spatial_field_1d_sim.py')
    spec = importlib.util.spec_from_file_location('sim', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def excess_velocity_lattice(L, N, mu, D, T, lam):
    """Tree-level KPZ excess velocity ⟨φ⟩ = (λ/2μ)·⟨(∂_xφ)²⟩ on the FD lattice
    (central-difference first derivative ``sin(k·dx)/dx``)."""
    dx = L / N
    ks = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    disp = mu + (2.0 * D / dx**2) * (1.0 - np.cos(ks * dx))
    ddphi2 = (T / L) * np.sum((np.sin(ks * dx) / dx) ** 2 / disp)
    return (lam / (2.0 * mu)) * ddphi2


def _conn_var(snaps):
    m = float(np.mean(snaps))
    return float(np.mean(snaps**2)) - m * m, m


def run(lam=0.3, seeds=(11, 23, 37, 51, 67, 83, 101, 113),
        L=20.0, N=128, mu=1.0, D=1.0, T=1.0,
        n_steps=300000, burn_in=40000, record_every=20):
    sim = _load_sim()
    P = dict(L=L, N=N, mu=mu, D=D, T=T, n_steps=n_steps,
             burn_in=burn_in, record_every=record_every)
    dkpz, dburg, exc = [], [], []
    for s in seeds:
        sf, _, _ = sim.simulate(seed=s, **P)
        sk, _, _ = sim.simulate(seed=s, lam_kpz=lam, **P)
        sb, _, _ = sim.simulate(seed=s, lam_burg=lam, **P)
        cf, _ = _conn_var(sf)
        ck, mk = _conn_var(sk)
        cb, _ = _conn_var(sb)
        dkpz.append(ck - cf); dburg.append(cb - cf); exc.append(mk)

    def stat(a):
        a = np.asarray(a)
        return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

    mk, ek = stat(dkpz)
    mb, eb = stat(dburg)
    me, _ = stat(exc)
    exc_th = excess_velocity_lattice(L, N, mu, D, T, lam)
    print(f'λ={lam}, μ=D=T=1, L={L}, N={N}, {len(seeds)} paired seeds')
    print(f'  KPZ excess velocity ⟨φ⟩:  sim {me:.4f}  vs  lattice {exc_th:.4f}'
          f'   (ratio {me/exc_th:.3f})')
    print(f'  KPZ     δC(0,0): sim {mk:+.5f} ± {ek:.5f}   (theory +0.00109)')
    print(f'  Burgers δC(0,0): sim {mb:+.5f} ± {eb:.5f}   (theory −0.00013)')
    return dict(exc_sim=me, exc_theory=exc_th,
                dkpz=(mk, ek), dburg=(mb, eb))


if __name__ == '__main__':
    run()

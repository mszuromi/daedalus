"""
Spectral exponential-Euler (ETD1) simulator for the 1D stochastic φ⁶
field (Allen-Cahn + quintic force) on a ring — the cross-check
simulator for ``allen_cahn_quintic_1d_subcritical_infinite``.

Integrates

    ∂_t φ(x,t) = -μ φ + D ∂_x² φ - λ φ³ - γ φ⁵ + η(x,t),
    ⟨η(x,t) η(x',t')⟩ = 2T δ(x - x') δ(t - t'),

on a uniform periodic grid of N points (dx = L/N).  The linear + noise
part is propagated EXACTLY per Fourier mode (an Ornstein-Uhlenbeck
update), so the λ=γ=0 stationary statistics are unbiased in dt; the
nonlinear force ``-λφ³ - γφ⁵`` is added through the ETD1 integrating
factor ``(1-e^{-ω dt})/ω``.

This is a near-copy of ``models/spatial_field_1d_sim.py`` specialized to
the φ³+φ⁵ force (the confining sextic potential
``V = μφ²/2 + λφ⁴/4 + γφ⁶/6`` is bounded below for γ ≥ 0, so the sim is
stable).  Returns the same ``(snapshots, x_grid, meta)`` API; the
``equal_time_correlator`` / ``lattice_sum_variance`` helpers match the
φ⁴ simulator so the notebook can be a drop-in.
"""
import numpy as np


def _dispersion(N, dx, mu, D):
    """Finite-difference lattice dispersion ω_k = μ + (2D/dx²)(1-cos)
    on the rfft grid (length N//2+1)."""
    M = N // 2 + 1
    m = np.arange(M)
    return mu + (2.0 * D / dx**2) * (1.0 - np.cos(2.0 * np.pi * m / N))


def _evolve(phi, n_steps, dt, mu, D, lam, gamma, T, dx, record_every, rng):
    """Spectral exponential-Euler (ETD1) integrator with the φ³+φ⁵ force."""
    N = phi.shape[0]
    L = N * dx
    n_rec = n_steps // record_every
    out = np.empty((n_rec, N), dtype=np.float64)

    omega = _dispersion(N, dx, mu, D)              # (M,)
    M = omega.shape[0]
    decay = np.exp(-omega * dt)
    etd1 = np.where(omega * dt > 1e-12, (1.0 - decay) / omega, dt)
    stat_var = T * N**2 / (L * omega)              # ⟨|φ̂_k|²⟩ stationary
    inc_std = np.sqrt((1.0 - decay**2) * stat_var)  # per-step OU increment

    # rfft real/complex structure: modes 0 and (if N even) N/2 are real.
    is_real_mode = np.zeros(M, dtype=bool)
    is_real_mode[0] = True
    if N % 2 == 0:
        is_real_mode[M - 1] = True

    a = np.fft.rfft(phi)
    ri = 0
    for step in range(n_steps):
        # Nonlinear force F = rfft(-λ φ³ - γ φ⁵).  The γφ⁵ term (→ +γφ⁶/6
        # potential) is the new confining sextic relative to pure φ⁴.
        if lam != 0.0 or gamma != 0.0:
            phi_r = np.fft.irfft(a, n=N)
            F = np.fft.rfft(-lam * phi_r**3 - gamma * phi_r**5)
        else:
            F = 0.0
        # OU noise increment with rfft Hermitian structure.
        noise = np.zeros(M, dtype=np.complex128)
        gr = rng.standard_normal(M)
        gi = rng.standard_normal(M)
        for mm in range(M):
            if is_real_mode[mm]:
                noise[mm] = inc_std[mm] * gr[mm]
            else:
                noise[mm] = inc_std[mm] / np.sqrt(2.0) * (gr[mm] + 1j * gi[mm])
        a = decay * a + etd1 * F + noise
        if (step + 1) % record_every == 0:
            out[ri, :] = np.fft.irfft(a, n=N)
            ri += 1
    return out


def simulate(L=20.0, N=200, mu=1.0, D=1.0, lam=0.1, gamma=0.1, T=1.0,
             dt=None, n_steps=400000, burn_in=40000, record_every=20,
             seed=12345):
    """Run the φ⁶ simulator and return ``(snapshots, x_grid, meta)``.

    snapshots : (n_rec, N) recorded field configurations (post burn-in)
    x_grid    : (N,) spatial coordinates
    meta      : dict with dx, dt, params, k_max (Nyquist cutoff)
    """
    # Cast counts/seed to Python ints (the SageMath kernel preparses int
    # literals into Sage Integer, which numpy's RNG path rejects).
    N = int(N)
    n_steps = int(n_steps)
    burn_in = int(burn_in)
    record_every = int(record_every)
    seed = int(seed)
    dx = L / N
    if dt is None:
        dt = min(0.02 / mu, 0.05) if (lam != 0.0 or gamma != 0.0) else 0.05
    rng = np.random.default_rng(seed)
    phi0 = np.zeros(N, dtype=np.float64)
    phi_burn = _evolve(phi0, burn_in, dt, mu, D, lam, gamma, T, dx, burn_in, rng)
    phi_start = phi_burn[-1, :].copy()
    snaps = _evolve(phi_start, n_steps, dt, mu, D, lam, gamma, T, dx,
                    record_every, rng)
    x_grid = np.arange(N) * dx
    meta = {'dx': dx, 'dt': dt, 'L': L, 'N': N, 'mu': mu, 'D': D,
            'lam': lam, 'gamma': gamma, 'T': T,
            'record_every': record_every, 'n_rec': snaps.shape[0],
            'k_max': np.pi / dx}
    return snaps, x_grid, meta


def structure_factor(snaps, meta):
    """Equal-time structure factor S(q) = ⟨|φ_q|²⟩ on the rfft grid,
    normalized as (L/N²)·⟨|FFT φ|²⟩ (tree → T/(μ+Dq²))."""
    L, N = meta['L'], int(meta['N'])
    q_grid = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    F = np.fft.rfft(snaps, axis=1)
    S = np.mean(np.abs(F) ** 2, axis=0) * (L / N ** 2)
    return q_grid, S


def equal_time_correlator(snaps):
    """C(x) = ⟨φ(x0) φ(x0 + x)⟩ via the periodic translational average.
    Returns C of length N (C[m] = correlator at separation m·dx)."""
    n_rec, N = snaps.shape
    acc = np.zeros(N)
    for r in range(n_rec):
        f = np.fft.rfft(snaps[r])
        ac = np.fft.irfft(np.abs(f) ** 2, n=N) / N
        acc += ac
    return acc / n_rec


def lattice_sum_variance(L, N, mu, D, T):
    """Discretized-theory exact equal-time variance ⟨φ²⟩ with the
    finite-difference dispersion — the simulator's λ=γ=0 reference."""
    N = int(N)
    dx = L / N
    ks = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    disp = mu + (2.0 * D / dx**2) * (1.0 - np.cos(ks * dx))
    return (T / L) * np.sum(1.0 / disp)
